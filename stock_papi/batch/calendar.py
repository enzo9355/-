"""以已驗證 TWSE 開休市 artifact 計算台股交易日。"""

import datetime
import re
from dataclasses import dataclass


TWSE_CALENDAR_URL = (
    "https://openapi.twse.com.tw/v1/holidaySchedule/holidaySchedule"
)
NYSE_CALENDAR_URL = "https://www.nyse.com/markets/hours-calendars"
VALID_CALENDAR_URLS = {
    "TW": {TWSE_CALENDAR_URL},
    "US": {NYSE_CALENDAR_URL, "https://www.sec.gov/edgar/searchedgar/companysearch"},
}


class CalendarError(ValueError):
    """交易日曆缺漏或不可信。"""


def _date(value, label):
    try:
        parsed = datetime.date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise CalendarError(f"{label} 日期不合法") from exc
    if parsed.isoformat() != value:
        raise CalendarError(f"{label} 日期不合法")
    return parsed


def _aware_datetime(value, label):
    try:
        text = str(value).replace("Z", "+00:00")
        text = re.sub(r"(\.\d{6})\d+(?=[+-]\d{2}:\d{2}$)", r"\1", text)
        parsed = datetime.datetime.fromisoformat(text)
    except (TypeError, ValueError) as exc:
        raise CalendarError(f"{label} 時間不合法") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CalendarError(f"{label} 必須包含時區")
    return parsed


def _date_set(values, year, label):
    if not isinstance(values, list):
        raise CalendarError(f"{label} 必須是清單")
    parsed = frozenset(_date(value, label) for value in values)
    if len(parsed) != len(values) or any(value.year != year for value in parsed):
        raise CalendarError(f"{label} 重複或超出年度")
    return parsed


@dataclass(frozen=True)
class TradingCalendar:
    market: str
    year: int
    source_url: str
    fetched_at: datetime.datetime
    source_sha256: str
    valid_from: datetime.date
    valid_to: datetime.date
    closed_dates: frozenset[datetime.date]
    special_open_dates: frozenset[datetime.date]
    early_closed_dates: frozenset[datetime.date] = frozenset()

    @classmethod
    def from_document(cls, document):
        if not isinstance(document, dict):
            raise CalendarError("calendar artifact 必須是 JSON object")
        market = document.get("market")
        year = document.get("year")
        source_url = document.get("source_url")
        source_sha256 = str(document.get("source_sha256") or "")
        if (
            document.get("schema_version") != 1
            or market not in ("TW", "US")
            or type(year) is not int
            or not 2000 <= year <= 2200
            or source_url not in VALID_CALENDAR_URLS.get(market, set())
            or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
        ):
            raise CalendarError("calendar artifact schema 不合法")
        valid_from = _date(document.get("valid_from"), "valid_from")
        valid_to = _date(document.get("valid_to"), "valid_to")
        if (
            valid_from != datetime.date(year, 1, 1)
            or valid_to != datetime.date(year, 12, 31)
        ):
            raise CalendarError("calendar artifact 未涵蓋完整年度")
        closed_dates = _date_set(document.get("closed_dates"), year, "closed_dates")
        special_open_dates = _date_set(
            document.get("special_open_dates", []), year, "special_open_dates"
        )
        early_closed_dates = _date_set(
            document.get("early_closed_dates", []), year, "early_closed_dates"
        )
        if closed_dates & special_open_dates:
            raise CalendarError("開市與休市日期不可重疊")
        if closed_dates & early_closed_dates:
            raise CalendarError("休市與提前收市日期不可重疊")
        return cls(
            market=market,
            year=year,
            source_url=source_url,
            fetched_at=_aware_datetime(document.get("fetched_at"), "fetched_at"),
            source_sha256=source_sha256,
            valid_from=valid_from,
            valid_to=valid_to,
            closed_dates=closed_dates,
            special_open_dates=special_open_dates,
            early_closed_dates=early_closed_dates,
        )

    def is_session(self, value):
        if not self.valid_from <= value <= self.valid_to:
            raise CalendarError("日期超出 calendar artifact 範圍")
        if value in self.special_open_dates:
            return True
        return value.weekday() < 5 and value not in self.closed_dates

    def is_early_close(self, value):
        if not self.valid_from <= value <= self.valid_to:
            raise CalendarError("日期超出 calendar artifact 範圍")
        return value in self.early_closed_dates


class TradingCalendarSet:
    def __init__(self, calendars):
        self._calendars = dict(calendars)

    @classmethod
    def from_documents(cls, documents):
        calendars = {}
        for document in documents:
            calendar = TradingCalendar.from_document(document)
            if calendar.year in calendars:
                raise CalendarError("同一年度 calendar artifact 重複")
            calendars[calendar.year] = calendar
        if not calendars:
            raise CalendarError("缺少 calendar artifact")
        return cls(calendars)

    def _calendar(self, value):
        calendar = self._calendars.get(value.year)
        if calendar is None:
            raise CalendarError(f"缺少 {value.year} 年 calendar artifact")
        return calendar

    def is_session(self, value):
        if not isinstance(value, datetime.date) or isinstance(value, datetime.datetime):
            raise CalendarError("交易日必須是 date")
        return self._calendar(value).is_session(value)

    def next_session(self, value):
        candidate = value + datetime.timedelta(days=1)
        while not self.is_session(candidate):
            candidate += datetime.timedelta(days=1)
        return candidate

    def session_offset(self, value, offset):
        if type(offset) is not int:
            raise CalendarError("交易日 offset 必須是整數")
        if not self.is_session(value):
            raise CalendarError("起始日期不是交易日")
        result = value
        direction = 1 if offset >= 0 else -1
        for _ in range(abs(offset)):
            result += datetime.timedelta(days=direction)
            while not self.is_session(result):
                result += datetime.timedelta(days=direction)
        return result

