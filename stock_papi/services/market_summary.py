"""Verified professional-report data adapted for market research pages."""

from reporting.professional_schema import ProfessionalPostCloseReport


def build_market_summary_view(report: ProfessionalPostCloseReport) -> dict:
    """Return only literal, verified fields needed by the market summary shell."""
    if not isinstance(report, ProfessionalPostCloseReport):
        raise TypeError("report must be ProfessionalPostCloseReport")

    return {
        "market": report.identity.market,
        "source_market_date": report.identity.source_market_date.isoformat(),
        "applicable_trading_date": report.identity.applicable_trading_date.isoformat(),
        "executive_summary": report.executive_summary.to_document(),
        "key_events": list(report.key_events),
        "market_observation": report.market.to_document(),
        "industries": report.industries.to_document(),
        "securities": report.securities.to_document(),
        "validation": report.validation.to_document(),
    }
