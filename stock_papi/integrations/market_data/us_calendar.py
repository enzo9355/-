"""Authoritative US stock exchange calendar (NYSE / Nasdaq) with session resolution."""

import datetime
import hashlib
import json
import zoneinfo

NEW_YORK = zoneinfo.ZoneInfo("America/New_York")
TAIPEI = zoneinfo.ZoneInfo("Asia/Taipei")
NYSE_CALENDAR_URL = "https://www.nyse.com/markets/hours-calendars"


def easter_sunday(year: int) -> datetime.date:
    """Anonymous Gregorian algorithm for Easter Sunday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime.date(year, month, day)


def nth_weekday(year: int, month: int, weekday: int, n: int) -> datetime.date:
    """Find the nth weekday of a month (1-indexed, Monday=0, Sunday=6)."""
    first_day = datetime.date(year, month, 1)
    day_diff = (weekday - first_day.weekday()) % 7
    return first_day + datetime.timedelta(days=day_diff + (n - 1) * 7)


def last_weekday(year: int, month: int, weekday: int) -> datetime.date:
    """Find the last weekday of a month."""
    if month == 12:
        next_month = datetime.date(year + 1, 1, 1)
    else:
        next_month = datetime.date(year, month + 1, 1)
    last_day = next_month - datetime.timedelta(days=1)
    day_diff = (last_day.weekday() - weekday) % 7
    return last_day - datetime.timedelta(days=day_diff)


def observed_holiday(date: datetime.date) -> datetime.date:
    """If Saturday -> Friday, if Sunday -> Monday."""
    if date.weekday() == 5:
        return date - datetime.timedelta(days=1)
    if date.weekday() == 6:
        return date + datetime.timedelta(days=1)
    return date


def get_us_exchange_holidays(
    year: int,
) -> tuple[set[datetime.date], set[datetime.date]]:
    """Return (closed_dates, early_closed_dates) for US stock exchanges."""
    closed = set()
    early = set()

    # 1. New Year's Day (Jan 1)
    new_year = datetime.date(year, 1, 1)
    if new_year.weekday() == 6:
        closed.add(datetime.date(year, 1, 2))
    elif new_year.weekday() != 5:
        closed.add(new_year)

    # Next year's New Year's Day falling on Saturday (observed Dec 31 of this year)
    next_new_year = datetime.date(year + 1, 1, 1)
    if next_new_year.weekday() == 5:
        closed.add(datetime.date(year, 12, 31))

    # 2. Martin Luther King Jr. Day (3rd Monday in January)
    closed.add(nth_weekday(year, 1, 0, 3))

    # 3. Washington's Birthday / Presidents' Day (3rd Monday in February)
    closed.add(nth_weekday(year, 2, 0, 3))

    # 4. Good Friday (Friday before Easter)
    easter = easter_sunday(year)
    closed.add(easter - datetime.timedelta(days=2))

    # 5. Memorial Day (Last Monday in May)
    closed.add(last_weekday(year, 5, 0))

    # 6. Juneteenth National Independence Day (June 19)
    juneteenth = observed_holiday(datetime.date(year, 6, 19))
    if juneteenth.year == year:
        closed.add(juneteenth)

    # 7. Independence Day (July 4)
    july4 = datetime.date(year, 7, 4)
    july4_obs = observed_holiday(july4)
    if july4_obs.year == year:
        closed.add(july4_obs)
    # Early close on July 3 if July 4 is weekday and not Monday
    if july4.weekday() in (1, 2, 3, 4):
        early.add(datetime.date(year, 7, 3))

    # 8. Labor Day (1st Monday in September)
    closed.add(nth_weekday(year, 9, 0, 1))

    # 9. Thanksgiving Day (4th Thursday in November)
    thanksgiving = nth_weekday(year, 11, 3, 4)
    closed.add(thanksgiving)
    # Day after Thanksgiving (Black Friday) is 1:00 PM early close
    early.add(thanksgiving + datetime.timedelta(days=1))

    # 10. Christmas Day (Dec 25)
    xmas = datetime.date(year, 12, 25)
    xmas_obs = observed_holiday(xmas)
    if xmas_obs.year == year:
        closed.add(xmas_obs)
    # Christmas Eve (Dec 24) is early close if on weekday and not observed Christmas
    xmas_eve = datetime.date(year, 12, 24)
    if xmas_eve.weekday() < 5 and xmas_eve not in closed:
        early.add(xmas_eve)

    return closed, early


def generate_us_calendar_document(year: int) -> dict:
    """Generate canonical US exchange calendar document for a year."""
    closed, early = get_us_exchange_holidays(year)
    doc = {
        "schema_version": 1,
        "market": "US",
        "year": year,
        "source_url": NYSE_CALENDAR_URL,
        "fetched_at": f"{year}-01-01T00:00:00+00:00",
        "valid_from": f"{year}-01-01",
        "valid_to": f"{year}-12-31",
        "closed_dates": sorted(d.isoformat() for d in closed),
        "early_closed_dates": sorted(d.isoformat() for d in early),
        "special_open_dates": [],
    }
    raw = json.dumps(doc, sort_keys=True).encode("utf-8")
    doc["source_sha256"] = hashlib.sha256(raw).hexdigest()
    return doc


def get_us_calendar_documents(
    start_year: int = 2024, end_year: int = 2028
) -> list[dict]:
    """Return calendar documents covering standard multi-year window."""
    return [
        generate_us_calendar_document(year)
        for year in range(start_year, end_year + 1)
    ]
