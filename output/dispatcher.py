"""
output/dispatcher.py
Main output dispatcher for ScoreBorga 2.5.

Orchestrates the full prediction pipeline:
  1. Fetch weekend fixtures from Sportmonks
  2. Fetch current odds from Odds API
  3. Run analytics per fixture
  4. Train or load ML model (for hybrid/ml modes)
  5. Generate predictions
  6. Polish predictions
  7. Send to Telegram
"""

import logging
from collections import Counter
from typing import List, Optional

from data.sportmonks import SportmonksClient
from data.odds_api import OddsApiClient
from engine.analytics import build_fixture_analytics
from engine.predictor import predict_all
from engine.polisher import polish_all
from output.telegram_bot import send_message
from leagues.top7 import TOP7_LEAGUES, LEAGUE_BY_ID
from config.settings import settings

logger = logging.getLogger(__name__)


def train_ml_model(
    sportmonks_client: Optional[SportmonksClient] = None,
    num_seasons: Optional[int] = None,
    league_ids: Optional[List[int]] = None,
) -> bool:
    """
    Train the ML model on historical data.

    Args:
        sportmonks_client: Optional Sportmonks client instance (creates new one if not provided)
        num_seasons: Number of seasons to train on (defaults to settings.HISTORICAL_SEASONS)
        league_ids: List of league IDs to fetch data for (defaults to settings.LEAGUE_IDS)

    Returns:
        True if training succeeded, False otherwise
    """
    from data.historical import HistoricalDataFetcher
    from engine.ml_model import get_ml_predictor

    client = sportmonks_client or SportmonksClient()
    num_seasons = num_seasons or settings.HISTORICAL_SEASONS
    league_ids = league_ids or settings.LEAGUE_IDS

    logger.info("=== Training ML model on %d seasons of historical data ===", num_seasons)

    # Fetch historical data
    fetcher = HistoricalDataFetcher(client)
    training_data = fetcher.fetch_historical_training_data(
        league_ids=league_ids,
        num_seasons=num_seasons,
    )

    if not training_data:
        logger.warning("No historical training data available")
        return False

    # Train the model
    predictor = get_ml_predictor()
    success = predictor.train(training_data)

    if success:
        # Save the trained model
        predictor.save_model()
        logger.info("=== ML model training complete ===")
    else:
        logger.error("=== ML model training failed ===")

    return success


def run_pipeline(
    dry_run: bool = False,
    league_ids: Optional[List[int]] = None,
    mode: Optional[str] = None,
    force_retrain: bool = False,
) -> str:
    """
    Execute the full prediction pipeline and optionally post results to Telegram.

    Args:
        dry_run: If True, skip sending to Telegram (returns the message string).
        league_ids: Optional list of Sportmonks league IDs to restrict to.
        mode: Prediction mode override ("stat", "ml", or "hybrid"). Defaults to settings.PREDICTION_MODE.
        force_retrain: If True, retrain the ML model even if one exists.

    Returns:
        The polished prediction message string.
    """
    mode = mode or settings.PREDICTION_MODE
    logger.info("=== ScoreBorga 2.5 pipeline starting (mode: %s) ===", mode)

    # 1. Fetch weekend fixtures
    sm_client = SportmonksClient()
    logger.info("Fetching weekend fixtures from Sportmonks...")
    fixtures = sm_client.get_weekend_fixtures(league_ids)
    logger.info("Found %d fixtures", len(fixtures))
    if fixtures:
        league_counts = Counter(f.get("league_id") for f in fixtures)
        logger.debug("Fixtures by league_id: %s", dict(league_counts))

    if not fixtures:
        logger.warning("No fixtures found for this weekend.")
        message = polish_all([])
        if not dry_run:
            send_message(message)
        return message

    # 2. Prepare ML model for hybrid/ml modes
    if mode in ("hybrid", "ml"):
        from engine.ml_model import get_ml_predictor
        predictor = get_ml_predictor()

        # Train if not already trained or if force_retrain is True
        if force_retrain or not predictor.is_trained:
            logger.info("ML model not trained, initiating training...")
            training_success = train_ml_model(sm_client, league_ids=league_ids)
            if not training_success:
                logger.warning("ML model training failed, falling back to stat mode")
                mode = "stat"
        else:
            logger.info("Using existing trained ML model")

    # 3. Fetch odds for all top-7 leagues
    odds_client = OddsApiClient()
    logger.info("Fetching odds from Odds API...")
    all_odds = odds_client.get_odds_for_all_top7(TOP7_LEAGUES)

    # Build a lookup: sport_key → list of events
    league_odds_map = {league["odds_key"]: all_odds.get(league["odds_key"], []) for league in TOP7_LEAGUES}

    # 4. Run analytics for each fixture
    analytics_list = []
    for fixture in fixtures:
        league_id = fixture.get("league_id")
        league_info = LEAGUE_BY_ID.get(league_id, {})
        sport_key = league_info.get("odds_key", "")

        home_participants = [p for p in fixture.get("participants", []) if p.get("meta", {}).get("location") == "home"]
        away_participants = [p for p in fixture.get("participants", []) if p.get("meta", {}).get("location") == "away"]

        home_id = home_participants[0].get("id") if home_participants else None
        away_id = away_participants[0].get("id") if away_participants else None
        home_name = home_participants[0].get("name", "") if home_participants else ""
        away_name = away_participants[0].get("name", "") if away_participants else ""

        # Fetch recent form and H2H
        home_recent: list = []
        away_recent: list = []
        h2h_fixtures: list = []
        try:
            if home_id:
                home_recent = sm_client.get_recent_fixtures(home_id)
            if away_id:
                away_recent = sm_client.get_recent_fixtures(away_id)
            if home_id and away_id:
                h2h_fixtures = sm_client.get_head_to_head(home_id, away_id)
        except Exception as exc:
            logger.warning("Could not fetch supplementary data for fixture %s: %s", fixture.get("id"), exc)

        # Map odds for this fixture
        odds_events = league_odds_map.get(sport_key, [])
        odds = odds_client.map_odds_to_fixture(odds_events, home_name, away_name)

        analytics = build_fixture_analytics(fixture, home_recent, away_recent, h2h_fixtures, odds)
        analytics_list.append(analytics)

    logger.info("Analytics built for %d fixtures", len(analytics_list))

    # 5. Generate predictions (with specified mode)
    predictions = predict_all(analytics_list, mode=mode)
    logger.info("Generated %d predictions", len(predictions))

    # 6. Polish predictions
    message = polish_all(predictions)

    # 7. Send to Telegram
    if not dry_run:
        logger.info("Sending predictions to Telegram...")
        success = send_message(message)
        if success:
            logger.info("Predictions sent successfully.")
        else:
            logger.error("Failed to send predictions to Telegram.")
    else:
        logger.info("Dry run — skipping Telegram dispatch.")

    logger.info("=== ScoreBorga 2.5 pipeline complete ===")
    return message
