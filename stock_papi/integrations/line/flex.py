import datetime

from stock_papi.shared.formatting import format_sentiment_summary as _format_sentiment_summary
from stock_papi.services.recommendation_engine import recommend_analysis


ABSORB_NAVY = "#122643"
ABSORB_INK = "#172033"
ABSORB_MUTED = "#5F6B7A"
ABSORB_SURFACE = "#FFFFFF"


def _observation_trend_label(value):
    return {
        "above_ma20_ma60": "站上 MA20 與 MA60",
        "above_ma20": "站上 MA20",
        "below_ma60": "低於 MA60",
        "mixed": "均線交錯",
    }.get(value, "資料不足")


def _stock_observation_bubble(code, name, body, url, watched):
    return {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": ABSORB_NAVY,
            "paddingAll": "20px",
            "contents": [
                {"type": "text", "text": "ABSORB｜市場觀察", "color": "#FFFFFF", "weight": "bold", "size": "xs"},
                {"type": "text", "text": f"{name} ({code})", "color": "#FFFFFF", "weight": "bold", "size": "xl", "wrap": True},
            ],
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#f8fafc",
            "paddingAll": "20px",
            "spacing": "md",
            "contents": body,
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#f8fafc",
            "paddingAll": "16px",
            "spacing": "sm",
            "contents": [
                {
                    "type": "button",
                    "style": "secondary",
                    "action": {
                        "type": "postback",
                        "label": "移除關注" if watched else "加入關注",
                        "data": f"watch:{'remove' if watched else 'add'}:{code}",
                    },
                },
                {
                    "type": "button",
                    "style": "secondary",
                    "action": {
                        "type": "postback",
                        "label": "設定實況提醒",
                        "data": f"alert:menu:{code}",
                    },
                },
                {
                    "type": "button",
                    "style": "primary",
                    "color": ABSORB_NAVY,
                    "action": {
                        "type": "uri",
                        "label": "查看完整觀察",
                        "uri": url,
                    },
                },
            ],
        },
    }


def build_stock_observation_flex(code, name, data, url, watched=False):
    """Render verified actual-market fields without model or backtest content."""
    if data.get("observation_kind") in {
        "officially_suspended",
        "official_no_regular_trade",
    }:
        body = [
            {
                "type": "text",
                "text": str(data["status_label"]),
                "color": "#b45309",
                "size": "md",
                "weight": "bold",
                "wrap": True,
            },
            {
                "type": "text",
                "text": f"官方狀態驗證日 {data['observation_as_of']}",
                "color": "#64748b",
                "size": "xs",
                "wrap": True,
            },
        ]
        if data.get("last_regular_close") is not None:
            body.append(
                {
                    "type": "text",
                    "text": (
                        "最後正常交易收盤 "
                        f"{float(data['last_regular_close']):.2f}"
                        f"（{data['latest_regular_price_date']}）"
                    ),
                    "color": "#64748b",
                    "size": "sm",
                    "wrap": True,
                }
            )
        return _stock_observation_bubble(code, name, body, url, watched)
    risk_events = [
        str(value)[:120]
        for value in data.get("risk_events", [])
        if isinstance(value, str) and value.strip()
    ][:3]
    body = [
        {
            "type": "text",
            "text": "AI 預測研究中",
            "color": "#b45309",
            "size": "sm",
            "weight": "bold",
            "wrap": True,
        },
        {
            "type": "text",
            "text": "目前只顯示已驗證的市場觀察資料。",
            "color": "#64748b",
            "size": "xs",
            "wrap": True,
        },
        {"type": "separator", "margin": "md", "color": "#cbd5e1"},
        {
            "type": "box",
            "layout": "horizontal",
            "contents": [
                {"type": "text", "text": "最新收盤", "color": "#64748b", "size": "sm", "flex": 4},
                {
                    "type": "text",
                    "text": f"{float(data['price']):.2f}",
                    "color": "#0f172a",
                    "size": "md",
                    "weight": "bold",
                    "align": "end",
                    "flex": 5,
                },
            ],
        },
        {
            "type": "box",
            "layout": "horizontal",
            "contents": [
                {"type": "text", "text": "均線狀態", "color": "#64748b", "size": "sm", "flex": 4},
                {
                    "type": "text",
                    "text": _observation_trend_label(data.get("trend_observation")),
                    "color": "#0f172a",
                    "size": "sm",
                    "weight": "bold",
                    "align": "end",
                    "wrap": True,
                    "flex": 5,
                },
            ],
        },
        {
            "type": "box",
            "layout": "horizontal",
            "contents": [
                {"type": "text", "text": "量比", "color": "#64748b", "size": "sm", "flex": 4},
                {
                    "type": "text",
                    "text": (
                        f"{float(data['volume_ratio']):.2f}"
                        if data.get("volume_ratio") is not None
                        else "資料不足"
                    ),
                    "color": "#0f172a",
                    "size": "sm",
                    "align": "end",
                    "flex": 5,
                },
            ],
        },
        {
            "type": "text",
            "text": f"資料日期 {data.get('as_of') or '待更新'}",
            "color": "#94a3b8",
            "size": "xs",
            "wrap": True,
        },
    ]
    if risk_events:
        body.extend(
            [
                {"type": "separator", "margin": "md", "color": "#cbd5e1"},
                {
                    "type": "text",
                    "text": "已觸發事件",
                    "color": "#0f172a",
                    "size": "sm",
                    "weight": "bold",
                },
                *[
                    {
                        "type": "text",
                        "text": f"• {event}",
                        "color": "#64748b",
                        "size": "xs",
                        "wrap": True,
                    }
                    for event in risk_events
                ],
            ]
        )
    return _stock_observation_bubble(code, name, body, url, watched)


def build_stock_flex_message(code, name, data, url, watched=False):
    color_prob = "#10b981" if data['prob'] >= 50 else "#ef4444"
    color_s = "#10b981" if data['s_score'] >= 50 else "#ef4444"
    color_trend = "#10b981" if "多" in data['trend'] else "#ef4444"
    sentiment_summary = _format_sentiment_summary(data)

    body_contents = []
    try:
        tz_taipei = datetime.timezone(datetime.timedelta(hours=8), "Asia/Taipei")
        today = datetime.datetime.now(tz_taipei).date()
        as_of_date = datetime.date.fromisoformat(data["as_of"])
        if (today - as_of_date).days >= 1:
            body_contents.append({
                "type": "box",
                "layout": "horizontal",
                "backgroundColor": "#fffbeb",
                "borderColor": "#fef3c7",
                "borderWidth": "1px",
                "cornerRadius": "6px",
                "paddingAll": "8px",
                "contents": [
                    {
                        "type": "text",
                        "text": f"⚠️ 數據延遲：此預測基於 {data['as_of']} 的市場數據。",
                        "color": "#b45309",
                        "size": "xs",
                        "wrap": True,
                        "weight": "bold"
                    }
                ]
            })
    except Exception:
        pass

    recommendation = data.get("recommendation")
    if not isinstance(recommendation, dict):
        try:
            recommendation = recommend_analysis(
                data,
                current_date=datetime.date.fromisoformat(str(data.get("as_of"))),
            ).to_dict()
        except (TypeError, ValueError):
            recommendation = recommend_analysis(data).to_dict()
    output_label = data.get("model_output_label") or "五日上漲機率"
    output_suffix = "%" if output_label == "五日上漲機率" else ""
    body_contents.extend([
        {
            "type": "text",
            "text": str(recommendation["action"]),
            "color": "#0f172a",
            "size": "xl",
            "weight": "bold",
            "wrap": True,
        },
        {
            "type": "text",
            "text": str(recommendation["headline"]),
            "color": "#475569",
            "size": "sm",
            "wrap": True,
        },
        {"type": "separator", "margin": "md", "color": "#cbd5e1"},
    ])

    body_contents.extend([
        {
            "type": "box",
            "layout": "horizontal",
            "contents": [
                { "type": "text", "text": "💰 最新收盤", "color": "#64748b", "size": "sm", "flex": 4 },
                { "type": "text", "text": f"{data['price']:.2f}", "color": "#0f172a", "size": "md", "weight": "bold", "align": "end", "flex": 5 }
            ]
        },
        {
            "type": "box",
            "layout": "horizontal",
            "contents": [
                { "type": "text", "text": "📈 當前趨勢", "color": "#64748b", "size": "sm", "flex": 4 },
                { "type": "text", "text": data['trend'], "color": color_trend, "size": "md", "weight": "bold", "align": "end", "flex": 5 }
            ]
        },
        {
            "type": "box",
            "layout": "horizontal",
            "contents": [
                { "type": "text", "text": "🌡 新聞／輿論情緒", "color": "#64748b", "size": "sm", "flex": 4, "wrap": True },
                { "type": "text", "text": f"{data['s_status']} ({data['s_score']:.1f})", "color": color_s, "size": "md", "weight": "bold", "align": "end", "flex": 5 }
            ]
        },
        {
            "type": "text",
            "text": sentiment_summary,
            "color": "#64748b",
            "size": "xs",
            "align": "end",
            "wrap": True
        },
        { "type": "separator", "margin": "md", "color": "#cbd5e1" },
        {
            "type": "box",
            "layout": "horizontal",
            "margin": "md",
            "contents": [
                { "type": "text", "text": f"🎯 {output_label}", "color": "#0f172a", "size": "md", "weight": "bold", "flex": 4 },
                { "type": "text", "text": f"{data['prob']}{output_suffix}", "color": color_prob, "size": "lg", "weight": "bold", "align": "end", "flex": 5 }
            ]
        },
        *([{
            "type": "text",
            "text": str(data["calibration_notice"]),
            "color": "#b45309",
            "size": "xs",
            "wrap": True,
        }] if data.get("calibration_notice") else [])
    ])

    return {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": ABSORB_NAVY,
            "paddingAll": "20px",
            "contents": [
                {
                    "type": "text", "text": "ABSORB", "color": "#FFFFFF",
                    "weight": "bold", "size": "xs",
                },
                {
                    "type": "text",
                    "text": f"📊 {name} ({code})",
                    "color": "#ffffff",
                    "weight": "bold",
                    "size": "xl"
                }
            ]
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#f8fafc",
            "paddingAll": "20px",
            "spacing": "md",
            "contents": body_contents
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#f8fafc",
            "paddingAll": "16px",
            "spacing": "sm",
            "contents": [
                {
                    "type": "button",
                    "style": "secondary",
                    "action": {
                        "type": "postback",
                        "label": "移除關注" if watched else "加入關注",
                        "data": f"watch:{'remove' if watched else 'add'}:{code}",
                    }
                },
                {
                    "type": "button",
                    "style": "secondary",
                    "action": {
                        "type": "postback",
                        "label": "設定提醒",
                        "data": f"alert:menu:{code}",
                    }
                },
                {
                    "type": "button",
                    "style": "secondary",
                    "action": {
                        "type": "postback",
                        "label": "投資試算",
                        "data": f"calc:menu:{code}",
                    }
                },
                {
                    "type": "button",
                    "style": "primary",
                    "color": ABSORB_NAVY,
                    "action": {
                        "type": "uri",
                        "label": "查看完整分析",
                        "uri": url,
                    }
                }
            ]
        }
    }


def _empty_line_bubble(title, description):
    return {
        "type": "bubble",
        "size": "kilo",
        "body": {
            "type": "box",
            "layout": "vertical",
            "paddingAll": "20px",
            "spacing": "md",
            "contents": [
                {"type": "text", "text": title, "weight": "bold", "size": "lg", "wrap": True},
                {"type": "text", "text": description, "color": "#64748b", "size": "sm", "wrap": True},
            ],
        },
    }


def _watchlist_card(item, snapshot, base_url):
    code = item["code"]
    name = item["name"]
    if snapshot:
        label = snapshot.get("model_output_label") or "五日上漲機率"
        suffix = "%" if label == "五日上漲機率" else ""
        details = [
            f"收盤價 {snapshot['price']:.2f}",
            f"{label} {snapshot['prob']}{suffix}",
            f"趨勢 {snapshot['trend']}",
            f"資料日期 {snapshot['as_of']}",
        ]
    else:
        details = ["待收盤更新"]
    return {
        "type": "bubble",
        "size": "kilo",
        "body": {
            "type": "box", "layout": "vertical", "paddingAll": "18px", "spacing": "sm",
            "contents": [
                {"type": "text", "text": f"{name} ({code})", "weight": "bold", "size": "lg", "wrap": True},
                *[
                    {"type": "text", "text": detail, "color": "#64748b", "size": "sm", "wrap": True}
                    for detail in details
                ],
            ],
        },
        "footer": {
            "type": "box", "layout": "vertical", "paddingAll": "14px", "spacing": "sm",
            "contents": [
                {"type": "button", "style": "secondary", "action": {
                    "type": "postback", "label": "移除關注", "data": f"watch:remove:{code}",
                }},
                {"type": "button", "style": "secondary", "action": {
                    "type": "postback", "label": "設定提醒", "data": f"alert:menu:{code}",
                }},
                {"type": "button", "style": "primary", "color": ABSORB_NAVY, "action": {
                    "type": "uri", "label": "查看完整分析",
                    "uri": f"{base_url.rstrip('/')}/stock/{code}",
                }},
            ],
        },
    }


def build_watchlist_flex(state, base_url):
    watchlist = state.get("watchlist", [])[:12]
    if not watchlist:
        return _empty_line_bubble("我的關注", "尚未加入關注股票。請先查詢個股，再點選「加入關注」。")
    snapshots = {
        item.get("code"): item
        for item in state.get("signals", {}).get("items", [])
        if isinstance(item, dict)
    }
    return {
        "type": "carousel",
        "contents": [
            _watchlist_card(item, snapshots.get(item.get("code")), base_url)
            for item in watchlist
        ],
    }


def build_observation_watchlist_flex(state, base_url):
    watchlist = [
        item for item in state.get("watchlist", [])
        if isinstance(item, dict) and item.get("code") and item.get("name")
    ][:12]
    if not watchlist:
        return _empty_line_bubble(
            "我的關注",
            "尚未加入關注股票。請先查詢個股，再點選「加入關注」。",
        )
    root = base_url.rstrip("/")
    return {
        "type": "carousel",
        "contents": [
            {
                "type": "bubble",
                "size": "kilo",
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "paddingAll": "18px",
                    "spacing": "sm",
                    "contents": [
                        {
                            "type": "text",
                            "text": f"{item['name']} ({item['code']})",
                            "weight": "bold",
                            "size": "lg",
                            "wrap": True,
                        },
                        {
                            "type": "text",
                            "text": "開啟個股頁查看最新已驗證觀察。",
                            "color": "#64748b",
                            "size": "sm",
                            "wrap": True,
                        },
                    ],
                },
                "footer": {
                    "type": "box",
                    "layout": "vertical",
                    "paddingAll": "14px",
                    "spacing": "sm",
                    "contents": [
                        {
                            "type": "button",
                            "style": "secondary",
                            "action": {
                                "type": "postback",
                                "label": "移除關注",
                                "data": f"watch:remove:{item['code']}",
                            },
                        },
                        {
                            "type": "button",
                            "style": "primary",
                            "color": ABSORB_NAVY,
                            "action": {
                                "type": "uri",
                                "label": "查看完整觀察",
                                "uri": f"{root}/stock/{item['code']}",
                            },
                        },
                    ],
                },
            }
            for item in watchlist
        ],
    }


def _alert_condition_text(alert):
    if alert["kind"] in {"price", "price_above"}:
        return f"收盤價站上 {float(alert['value']):g}"
    if alert["kind"] == "price_below":
        return f"收盤價跌破 {float(alert['value']):g}"
    if alert["kind"] == "probability":
        return f"模型輸出達到 {float(alert['value']):g}"
    return f"趨勢為{alert['value']}"


def _alert_management_card(alert):
    return {
        "type": "bubble",
        "size": "kilo",
        "body": {
            "type": "box", "layout": "vertical", "paddingAll": "18px", "spacing": "sm",
            "contents": [
                {"type": "text", "text": "提醒管理", "weight": "bold", "size": "sm", "color": ABSORB_NAVY},
                {"type": "text", "text": f"{alert['name']} ({alert['code']})", "weight": "bold", "size": "lg", "wrap": True},
                {"type": "text", "text": _alert_condition_text(alert), "color": "#64748b", "size": "sm", "wrap": True},
            ],
        },
        "footer": {
            "type": "box", "layout": "vertical", "paddingAll": "14px",
            "contents": [{"type": "button", "style": "secondary", "action": {
                "type": "postback", "label": "取消提醒", "data": f"alert:remove:{alert['id']}",
            }}],
        },
    }


def build_alerts_flex(state, prediction_allowed=True):
    alerts = [
        alert for alert in state.get("alerts", [])
        if alert.get("enabled", True)
        and (prediction_allowed or alert.get("kind") != "probability")
    ][:12]
    if not alerts:
        return _empty_line_bubble("提醒管理", "尚未設定提醒。請先查詢個股，再點選「設定提醒」。")
    return {"type": "carousel", "contents": [_alert_management_card(alert) for alert in alerts]}


def build_alert_menu_flex(code, name, prediction_allowed=True):
    choices = [
        ("站上收盤價", f"alert:start:{code}:price_above"),
        ("跌破收盤價", f"alert:start:{code}:price_below"),
        ("趨勢為多頭", f"alert:trend:{code}:多頭"),
        ("趨勢為空頭", f"alert:trend:{code}:空頭"),
    ]
    if prediction_allowed:
        choices.insert(
            2,
            ("上漲機率門檻", f"alert:start:{code}:probability"),
        )
    return {
        "type": "bubble",
        "size": "kilo",
        "body": {
            "type": "box", "layout": "vertical", "paddingAll": "18px", "spacing": "sm",
            "contents": [
                {"type": "text", "text": f"設定 {name} ({code}) 提醒", "weight": "bold", "size": "lg", "wrap": True},
                *[
                    {"type": "button", "style": "secondary", "action": {
                        "type": "postback", "label": label, "data": payload,
                    }}
                    for label, payload in choices
                ],
            ],
        },
    }


def build_calculator_menu_flex(code, name):
    choices = [("1 萬", 10000), ("5 萬", 50000), ("10 萬", 100000)]
    return {
        "type": "bubble",
        "size": "kilo",
        "body": {
            "type": "box", "layout": "vertical", "paddingAll": "18px", "spacing": "sm",
            "contents": [
                {"type": "text", "text": f"{name} 投資試算", "weight": "bold", "size": "lg", "wrap": True},
                {"type": "text", "text": "請選擇投入金額，或點自訂金額查看輸入格式。", "color": "#64748b", "size": "sm", "wrap": True},
                *[
                    {"type": "button", "style": "secondary", "action": {
                        "type": "postback", "label": label, "data": f"calc:amount:{code}:{amount}",
                    }}
                    for label, amount in choices
                ],
                {"type": "button", "style": "secondary", "action": {
                    "type": "postback", "label": "自訂金額", "data": f"calc:custom:{code}",
                }},
            ],
        },
    }


def _signal_card(item, base_url):
    code = item["code"]
    label = item.get("model_output_label") or "五日上漲機率"
    suffix = "%" if label == "五日上漲機率" else ""
    return {
        "type": "bubble",
        "size": "kilo",
        "body": {
            "type": "box", "layout": "vertical", "paddingAll": "18px", "spacing": "sm",
            "contents": [
                {"type": "text", "text": f"{item['name']} ({code})", "weight": "bold", "size": "lg", "wrap": True},
                {"type": "text", "text": f"收盤價 {item['price']:.2f}", "color": "#64748b", "size": "sm"},
                {"type": "text", "text": f"{label} {item['prob']}{suffix}", "color": "#64748b", "size": "sm"},
                {"type": "text", "text": f"趨勢 {item['trend']}", "color": "#64748b", "size": "sm"},
                {"type": "text", "text": f"資料日期 {item['as_of']}", "color": "#64748b", "size": "sm"},
            ],
        },
        "footer": {
            "type": "box", "layout": "vertical", "paddingAll": "14px",
            "contents": [{"type": "button", "style": "primary", "color": ABSORB_NAVY, "action": {
                "type": "uri", "label": "查看完整分析",
                "uri": f"{base_url.rstrip('/')}/stock/{code}",
            }}],
        },
    }


def build_strong_signals_flex(state, base_url):
    items = state.get("signals", {}).get("items", [])[:5]
    if not items:
        return _empty_line_bubble("強勢訊號", "尚無最新強勢訊號，請等待下一次收盤更新。")
    return {
        "type": "carousel",
        "contents": [_signal_card(item, base_url) for item in items],
    }


def build_alert_push_flex(hits, base_url):
    if not 1 <= len(hits) <= 12:
        raise ValueError("LINE Flex carousel requires 1 to 12 bubbles")

    def bubble(hit):
        alert, quote = hit["alert"], hit["quote"]
        if alert["kind"] in {"price", "price_above"}:
            condition = f"條件：收盤價站上 {float(alert['value']):g}"
            current = f"今日收盤價：{quote['price']:.2f}"
        elif alert["kind"] == "price_below":
            condition = f"條件：收盤價跌破 {float(alert['value']):g}"
            current = f"今日收盤價：{quote['price']:.2f}"
        elif alert["kind"] == "probability":
            label = quote.get("model_output_label") or "模型輸出"
            condition = f"條件：{label}達到 {float(alert['value']):g}"
            current = f"目前{label}：{quote['prob']}"
        else:
            condition = f"條件：趨勢為{alert['value']}"
            current = f"目前趨勢：{quote['trend']}"
        return {
            "type": "bubble", "size": "kilo",
            "header": {
                "type": "box", "layout": "vertical", "backgroundColor": ABSORB_NAVY,
                "paddingAll": "16px", "contents": [{
                    "type": "text", "text": "ABSORB 股票提醒", "color": "#FFFFFF",
                    "weight": "bold", "size": "sm",
                }],
            },
            "body": {
                "type": "box", "layout": "vertical", "paddingAll": "18px", "spacing": "sm",
                "contents": [
                    {"type": "text", "text": f"{quote['name']} ({quote['code']})", "weight": "bold", "size": "lg", "wrap": True},
                    {"type": "text", "text": condition, "size": "sm", "color": "#64748b", "wrap": True},
                    {"type": "text", "text": current, "size": "sm", "color": "#0f766e", "weight": "bold", "wrap": True},
                    {"type": "text", "text": f"資料日期：{quote['as_of']}", "size": "xs", "color": "#94a3b8"},
                ],
            },
            "footer": {
                "type": "box", "layout": "vertical", "paddingAll": "14px",
                "contents": [{"type": "button", "style": "primary", "color": ABSORB_NAVY, "action": {
                    "type": "uri", "label": "查看完整分析",
                    "uri": f"{base_url.rstrip('/')}/stock/{quote['code']}",
                }}],
            },
        }

    return {"type": "carousel", "contents": [bubble(hit) for hit in hits]}


def build_line_summary_card(title, lines, cta_label, url, accent=ABSORB_NAVY, action=None):
    """建立只有一個主要動作的 LINE 摘要卡。"""
    action = action or {"type": "uri", "label": cta_label, "uri": url}
    return {
        "type": "bubble",
        "size": "kilo",
        "header": {
            "type": "box", "layout": "vertical", "backgroundColor": ABSORB_NAVY,
            "paddingAll": "16px", "contents": [{
                "type": "text", "text": "ABSORB", "color": "#FFFFFF",
                "size": "xs", "weight": "bold",
            }],
        },
        "body": {
            "type": "box", "layout": "vertical", "backgroundColor": ABSORB_SURFACE,
            "paddingAll": "18px", "spacing": "md", "contents": [
                {"type": "text", "text": title, "color": ABSORB_INK, "size": "lg", "weight": "bold", "wrap": True},
                *[{"type": "text", "text": line, "color": ABSORB_MUTED, "size": "sm", "wrap": True} for line in lines],
            ],
        },
        "footer": {
            "type": "box", "layout": "vertical", "backgroundColor": ABSORB_SURFACE,
            "paddingAll": "14px", "contents": [{
                "type": "button", "style": "primary", "color": accent,
                "action": action,
            }],
        },
    }


def build_line_navigation_flex(base_url):
    """Rich Menu 入口的可預覽 Flex 版本。"""
    root = base_url.rstrip("/")
    entries = [
        ("看大盤", "查看市場報酬、廣度與風險狀態", "查看市場", {"type": "uri", "label": "查看市場", "uri": f"{root}/market"}),
        ("看產業", "查看產業實際報酬與市場廣度", "查看產業", {"type": "uri", "label": "查看產業", "uri": f"{root}/industries"}),
        ("查自選", "自選股票清單", "開啟關注", {"type": "message", "label": "開啟關注", "text": "我的關注"}),
        ("設提醒", "管理收盤價與均線趨勢通知", "管理提醒", {"type": "message", "label": "管理提醒", "text": "提醒管理"}),
        ("查股票", "輸入股票代碼查看實際資料", "查台積電", {"type": "message", "label": "查台積電", "text": "2330"}),
        ("市場觀察", "查看完整市場與事件頁面", "開啟觀察", {"type": "uri", "label": "開啟觀察", "uri": f"{root}/dashboard"}),
    ]
    return {
        "type": "carousel",
        "contents": [build_line_summary_card(title, [description], cta, root, action=action) for title, description, cta, action in entries],
    }


def build_calculator_help_flex():
    return build_line_summary_card(
        "投資試算",
        [
            "先輸入股票代碼，例如 2330。",
            "查詢結果會出現「投資試算」按鈕，可直接選 1 萬 / 5 萬 / 10 萬。",
            "自訂金額可輸入：試算 2330 100000",
        ],
        "輸入 2330 開始",
        "2330",
        action={"type": "message", "label": "輸入 2330 開始", "text": "2330"},
    )


def build_welcome_flex():
    return {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": ABSORB_NAVY,
            "paddingAll": "20px",
            "contents": [
                { "type": "text", "text": "ABSORB", "color": "#FFFFFF", "weight": "bold", "size": "xl" },
                { "type": "text", "text": "已驗證市場觀察", "color": "#DCE6F2", "size": "sm", "wrap": True }
            ]
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": ABSORB_SURFACE,
            "paddingAll": "20px",
            "spacing": "md",
            "contents": [
                { "type": "text", "text": "AI 預測研究中；目前正式服務只呈現已驗證的市場實況。", "color": ABSORB_INK, "size": "md", "weight": "bold", "wrap": True },
                { "type": "text", "text": "您可以：\n1️⃣ 開啟產業實際強弱頁\n2️⃣ 直接輸入股票代碼（如 2330）\n3️⃣ 輸入「大盤」查看市場實況\n4️⃣ 管理關注與收盤價提醒", "color": ABSORB_MUTED, "size": "sm", "wrap": True, "margin": "md" }
            ]
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": ABSORB_SURFACE,
            "paddingAll": "16px",
            "contents": [
                {
                    "type": "button",
                    "style": "primary",
                    "color": ABSORB_NAVY,
                    "action": { "type": "message", "label": "🎓 新手怎麼看？(教學)", "text": "新手教學" }
                }
            ]
        }
    }


def build_tutorial_flex():
    return {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": ABSORB_NAVY,
            "paddingAll": "20px",
            "contents": [
                { "type": "text", "text": "ABSORB 新手快速上手指南", "color": "#ffffff", "weight": "bold", "size": "xl", "wrap": True }
            ]
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": ABSORB_SURFACE,
            "paddingAll": "20px",
            "spacing": "md",
            "contents": [
                { "type": "text", "text": "正式服務只呈現已發生的市場資料，可先掌握以下三個重點：", "color": "#475569", "size": "sm", "wrap": True, "weight": "bold", "margin": "sm" },
                { "type": "separator", "margin": "md", "color": "#cbd5e1" },
                { "type": "text", "text": "1. 看「市場廣度」", "color": "#0f172a", "size": "md", "weight": "bold", "margin": "md" },
                { "type": "text", "text": "上漲家數與站上均線比例，可觀察漲勢是否由多數股票共同參與。", "color": "#64748b", "size": "sm", "wrap": True },
                { "type": "text", "text": "2. 看「相對報酬」", "color": "#0f172a", "size": "md", "weight": "bold", "margin": "md" },
                { "type": "text", "text": "產業報酬扣除同期大盤報酬，只描述已發生的相對強弱。", "color": "#64748b", "size": "sm", "wrap": True },
                { "type": "text", "text": "3. 看「異常事件」", "color": "#0f172a", "size": "md", "weight": "bold", "margin": "md" },
                { "type": "text", "text": "價格、量能、技術或資料品質超過條件時才列出；事件本身不是買賣指令。", "color": "#64748b", "size": "sm", "wrap": True }
            ]
        }
    }
