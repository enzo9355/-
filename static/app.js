const bySelector = (selector, root = document) => root.querySelector(selector);

function migrateLegacyHashRoute() {
  if (!["/", "/dashboard"].includes(window.location.pathname)) return;
  const target = {
    "#market-pulse": "/market",
    "#daily-focus": "/",
    "#market-heatmap": "/industries",
    "#industry-observations": "/industries",
    "#stock-search": "/stocks",
    "#stock-events": "/stocks",
    "#etf-observations": "/stocks?tab=etf",
    "#learn": "/learn",
  }[window.location.hash];
  if (target) window.location.replace(target);
}

function element(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined && text !== null) item.textContent = String(text);
  return item;
}

function replaceContent(container, items) {
  container.replaceChildren(...items);
}

function emptyState(message) {
  return element("div", "empty-state", message);
}

function stockHref(code, fallback = "/dashboard") {
  const normalized = String(code || "").toUpperCase();
  return /^[A-Z0-9.-]{1,16}$/.test(normalized)
    ? `/stock/${encodeURIComponent(normalized)}`
    : fallback;
}

function card(tag, className, rows, href) {
  const item = element(tag, className);
  if (href) item.setAttribute("href", href);
  rows.forEach(([rowTag, rowClass, value]) => item.append(element(rowTag, rowClass, value)));
  return item;
}

async function loadDashboard() {
  const page = bySelector("[data-dashboard-endpoint]");
  if (!page) return;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch(page.dataset.dashboardEndpoint, {
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });
    if (!response.ok) throw new Error("dashboard");
    renderDashboard(await response.json());
  } catch (_error) {
    const banner = bySelector("[data-dashboard-error]");
    if (banner) banner.hidden = false;
  } finally {
    window.clearTimeout(timeout);
  }
}

function displayNumber(value, digits = 2, suffix = "") {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(digits)}${suffix}` : "資料不足";
}

function displaySigned(value, digits = 2, suffix = "%") {
  const number = Number(value);
  if (!Number.isFinite(number)) return "資料不足";
  return `${number >= 0 ? "+" : ""}${number.toFixed(digits)}${suffix}`;
}

function observationTrend(value) {
  return {
    above_ma20_ma60: "站上 MA20 與 MA60",
    above_ma20: "站上 MA20",
    below_ma60: "低於 MA60",
    mixed: "均線交錯",
    insufficient: "資料不足",
  }[value] || "資料不足";
}

function renderDashboard(data) {
  if (!data || data.product_mode !== "observation") {
    throw new Error("observation dashboard required");
  }
  const marketData = data.market_observation || {};
  const hero = bySelector("[data-market-hero]");
  if (hero) {
    const riskState = {
      normal: "一般",
      cautious: "謹慎",
      elevated: "升高",
    }[marketData.risk_state] || "資料不足";
    replaceContent(hero, [
      element("p", "action-label action-insufficient", `市場風險狀態：${riskState}`),
      element("h1", "market-headline", `上漲 ${marketData.advancing_count ?? "—"} 檔、下跌 ${marketData.declining_count ?? "—"} 檔`),
      element("p", "hero-risk", `近 5 日市場中位報酬 ${displaySigned(marketData.return_5d_pct)}`),
      element("p", "muted small", `資料日 ${data.observation_as_of || "待更新"} · 覆蓋率 ${displayNumber((data.data_quality?.coverage || 0) * 100, 1, "%")}`),
    ]);
  }

  const market = bySelector("[data-market-summary]");
  if (market) {
    replaceContent(market, [
      card("article", "pulse-card", [["span", "", "單日中位報酬"], ["strong", "", displaySigned(marketData.return_1d_pct)], ["small", "muted", "全市場有效樣本"]]),
      card("article", "pulse-card", [["span", "", "站上 MA20"], ["strong", "", displayNumber(marketData.ma20_breadth_pct, 1, "%")], ["small", "muted", "市場均線廣度"]]),
      card("article", "pulse-card", [["span", "", "20 日已實現波動"], ["strong", "", displayNumber(marketData.realized_volatility_20d_pct, 1, "%")], ["small", "muted", `20 日新高 ${marketData.new_high_20d_count ?? "—"}／新低 ${marketData.new_low_20d_count ?? "—"}`]]),
    ]);
  }
  const status = bySelector(".status-dot");
  if (status) status.textContent = data.observation_as_of ? `資料日 ${data.observation_as_of}` : "資料不足";

  const focus = bySelector("[data-daily-focus]");
  if (focus) {
    const items = Array.isArray(data.daily_focus) ? data.daily_focus : [];
    replaceContent(focus, items.length ? items.map((text, index) =>
      card("article", "focus-card", [["span", "", `焦點 ${index + 1}`], ["strong", "", text]])
    ) : [emptyState("今日焦點資料不足。")]);
  }

  const heatmap = bySelector("[data-market-heatmap]");
  if (heatmap) {
    const cells = data.heatmap || [];
    if (cells.length > 0 && cells.length < 3) {
      heatmap.style.gridTemplateColumns = `repeat(${cells.length}, minmax(0, 1fr))`;
    } else {
      heatmap.style.gridTemplateColumns = "";
    }
    replaceContent(heatmap, cells.length ? cells.map((item) =>
      card("a", `heatmap-cell ${["hot", "cold", "steady"].includes(item.tone) ? item.tone : "steady"}`, [
        ["span", "", item.name],
        ["strong", "", displaySigned(item.metric_value_pct)],
        ["small", "", `${item.available_count ?? "—"} 檔 · 覆蓋 ${displayNumber((item.coverage || 0) * 100, 1, "%")}`],
      ], "/industries")
    ) : [emptyState("產業相對報酬資料不足。")]);
  }

  const industries = bySelector("[data-industry-observations]");
  if (industries) {
    const items = Array.isArray(data.industry_observations) ? data.industry_observations : [];
    replaceContent(industries, items.length ? items.map((item) =>
      card("article", "forecast-card", [
        ["span", "", item.name],
        ["strong", "", `相對大盤 ${displaySigned(item.relative_return_5d_pct)}`],
        ["small", "", `單日 ${displaySigned(item.return_1d_pct)} · 5 日 ${displaySigned(item.return_5d_pct)}`],
        ["small", "", `上漲家數 ${displayNumber(item.advancing_ratio_pct, 1, "%")} · 站上 MA20 ${displayNumber(item.ma20_breadth_pct, 1, "%")}`],
        ["small", "", `量比中位數 ${displayNumber(item.median_volume_ratio)} · 可用 ${item.available_count ?? "—"}/${item.component_count ?? "—"} 檔`],
      ])
    ) : [emptyState("產業觀察資料不足。")]);
  }

  const events = bySelector("[data-stock-events]");
  if (events) {
    const items = Array.isArray(data.stock_events) ? data.stock_events : [];
    replaceContent(events, items.length ? items.map((item) =>
      card("a", "pick-card", [
        ["span", "badge-stock", `${item.name} · ${item.symbol}`],
        ["strong", "", item.observation],
        ["p", "", `${item.metric_value ?? "—"} ${item.unit || ""}`],
        ["small", "", `資料日 ${item.as_of || data.observation_as_of}`],
      ], stockHref(item.symbol))
    ) : [emptyState("目前沒有通過條件的異常事件。")]);
  }

  const etfs = bySelector("[data-etf-observations]");
  if (etfs) {
    const items = Array.isArray(data.etf_observations) ? data.etf_observations : [];
    replaceContent(etfs, items.length ? items.map((item) =>
      card("a", "pick-card", [
        ["span", "badge-etf", `${item.name} · ${item.symbol}`],
        ["strong", "", `收盤 ${displayNumber(item.price)}`],
        ["p", "", `單日 ${displaySigned(item.return_1d_pct)} · 5 日 ${displaySigned(item.return_5d_pct)}`],
        ["small", "", `${observationTrend(item.trend_observation)} · 量比 ${displayNumber(item.volume_ratio)}`],
      ], stockHref(item.symbol))
    ) : [emptyState("ETF 觀察資料不足。")]);
  }
}

function loginLocation() {
  const returnTo = `${window.location.pathname}${window.location.search}`;
  return `/auth/line/login?return_to=${encodeURIComponent(returnTo)}`;
}

function updateAccountInterface(data) {
  window.absorbAccount = data;
  const account = bySelector("[data-account-nav]");
  if (account) {
    const profile = element("a", "account-profile-link");
    profile.href = "/account";
    if (data.user.picture_url) {
      const picture = element("img", "account-avatar");
      picture.src = data.user.picture_url;
      picture.alt = "";
      picture.referrerPolicy = "no-referrer";
      profile.append(picture);
    }
    const label = element("span", "");
    label.append(element("strong", "", data.user.display_name));
    label.append(element("small", "", "已連結 LINE"));
    profile.append(label);
    const watchlist = element("a", "nav-link", `我的關注（${data.watchlist.length}）`);
    watchlist.href = "/account/watchlist";
    const logout = element("button", "link-button", "登出");
    logout.type = "button";
    logout.dataset.accountLogout = "";
    replaceContent(account, [profile, watchlist, logout]);
  }
  const mobile = bySelector("[data-mobile-account]");
  if (mobile) mobile.textContent = "我的";
  const toggle = bySelector("[data-watchlist-toggle]");
  if (toggle) {
    const watched = data.watchlist.some((item) => item.code === toggle.dataset.code);
    toggle.dataset.authenticated = "true";
    toggle.dataset.watched = String(watched);
    toggle.textContent = watched ? "取消關注" : "加入關注";
    toggle.setAttribute("aria-pressed", String(watched));
  }
}

async function loadAccountState() {
  if (document.body.dataset.accountSession !== "present") return;
  try {
    const response = await fetch("/api/account/state", { headers: { Accept: "application/json" } });
    if (!response.ok) return;
    updateAccountInterface(await response.json());
  } catch (_error) {
    // 公開頁維持未登入狀態；不顯示內部錯誤。
  }
}

function appendConversationMessage(log, role, text) {
  const empty = bySelector(".empty-state", log);
  if (empty) empty.remove();
  const message = element("p", `conversation-message ${role}`, text);
  log.append(message);
  log.scrollTop = log.scrollHeight;
}

function initConversations() {
  document.querySelectorAll("[data-conversation-form]").forEach((form) => {
    const panel = form.closest("[data-conversation-endpoint]");
    const log = panel && bySelector("[data-conversation-log]", panel);
    if (!panel || !log) return;
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const input = bySelector("input[name='question']", form);
      const button = bySelector("button", form);
      const question = input.value.trim();
      if (!question) return;
      appendConversationMessage(log, "user", question);
      input.value = "";
      button.disabled = true;
      const headers = { Accept: "application/json", "Content-Type": "application/json" };
      if (window.absorbAccount?.csrf_token) headers["X-CSRF-Token"] = window.absorbAccount.csrf_token;
      try {
        const response = await fetch(panel.dataset.conversationEndpoint, {
          method: "POST",
          headers,
          body: JSON.stringify({
            question,
            market: panel.dataset.marketContext,
            page: panel.dataset.pageContext,
          }),
        });
        const data = await response.json();
        appendConversationMessage(log, "assistant", response.ok ? data.text : "自然語言分析暫時無法使用，請稍後再試。");
      } catch (_error) {
        appendConversationMessage(log, "assistant", "自然語言分析暫時無法使用，固定指令與股票查詢不受影響。");
      } finally {
        button.disabled = false;
        input.focus();
      }
    });
  });
}

function initQuickAsk() {
  const dialog = bySelector("[data-quick-ask-dialog]");
  const dataQuickAskOpen = document.querySelectorAll("[data-quick-ask-open]");
  if (!dialog || !dataQuickAskOpen.length) return;
  const closeButton = bySelector("[data-quick-ask-close]", dialog);
  const input = bySelector("input[name='question']", dialog);
  let lastFocus = null;

  const open = (instant = false) => {
    if (!dialog.hidden) {
      input.focus({ preventScroll: true });
      return;
    }
    lastFocus = document.activeElement;
    dialog.hidden = false;
    dataQuickAskOpen.forEach((button) => button.setAttribute("aria-expanded", "true"));
    const reveal = () => {
      dialog.classList.add("is-open");
      input.focus({ preventScroll: true });
    };
    if (instant) reveal();
    else window.requestAnimationFrame(reveal);
  };

  const close = () => {
    dialog.classList.remove("is-open");
    dataQuickAskOpen.forEach((button) => button.setAttribute("aria-expanded", "false"));
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    window.setTimeout(() => {
      dialog.hidden = true;
      if (lastFocus instanceof HTMLElement) lastFocus.focus({ preventScroll: true });
    }, reduceMotion ? 0 : 180);
  };

  dataQuickAskOpen.forEach((button) => button.addEventListener("click", () => open(false)));
  closeButton.addEventListener("click", close);
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) close();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !dialog.hidden) close();
    if (event.key === "Tab" && !dialog.hidden) {
      const focusable = [...dialog.querySelectorAll(
        'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
      )];
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (!first || !last) {
        event.preventDefault();
      } else if (event.shiftKey && (document.activeElement === first || !dialog.contains(document.activeElement))) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      open(true);
    }
  });
}

async function toggleWatchlist(button) {
  const account = window.absorbAccount;
  if (!account) {
    window.location.assign(loginLocation());
    return;
  }
  button.disabled = true;
  try {
    const response = await fetch("/api/account/watchlist", {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
        "X-CSRF-Token": account.csrf_token,
      },
      body: JSON.stringify({
        action: button.dataset.watched === "true" ? "remove" : "add",
        code: button.dataset.code,
      }),
    });
    if (!response.ok) throw new Error("watchlist");
    account.watchlist = (await response.json()).watchlist;
    updateAccountInterface(account);
  } catch (_error) {
    button.textContent = "暫時無法更新，請稍後再試";
  } finally {
    button.disabled = false;
  }
}

function formatNumber(value) {
  return Number.isFinite(value) ? Math.round(value).toLocaleString("zh-TW") : "—";
}

function initReturnCalculator() {
  const panel = bySelector("[data-return-calculator]");
  if (!panel) return;
  const input = bySelector("[data-investment-amount]", panel);
  const price = Number(panel.dataset.price);
  const strategyReturn = Number(panel.dataset.strategyReturn);
  const buyholdReturn = Number(panel.dataset.buyholdReturn);
  const update = () => {
    const amount = Number(input.value);
    const shares = Math.floor(amount / price);
    const deployed = shares * price;
    const valid = Number.isFinite(amount) && amount > 0 && price > 0 && shares > 0;
    bySelector("[data-shares]", panel).textContent = valid ? shares.toLocaleString("zh-TW") : "—";
    bySelector("[data-deployed]", panel).textContent = valid ? formatNumber(deployed) : "—";
    bySelector("[data-strategy-profit]", panel).textContent = valid ? formatNumber((deployed * strategyReturn) / 100) : "—";
    bySelector("[data-buyhold-profit]", panel).textContent = valid ? formatNumber((deployed * buyholdReturn) / 100) : "—";
  };
  input.addEventListener("input", update);
  update();
}

function measureChartHeight(container) {
  return Math.max(320, Math.min(460, Math.round(container.clientWidth * 0.62)));
}

function setChartRange(days) {
  if (!window.stockChart) return;
  const { chart, length } = window.stockChart;
  chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, length - days), to: length + 5 });
}

function initStockChart() {
  const container = bySelector("#stock-chart");
  const source = bySelector("#stock-chart-data");
  if (!container || !source || !window.LightweightCharts) return;
  const raw = JSON.parse(source.textContent);
  const candles = JSON.parse(raw.candles);
  const height = measureChartHeight(container);
  const chart = LightweightCharts.createChart(container, {
    width: container.clientWidth,
    height,
    layout: { background: { color: "transparent" }, textColor: "#586579" },
    grid: { vertLines: { color: "#d9e0e8" }, horzLines: { color: "#d9e0e8" } },
    timeScale: { borderColor: "#b7c1cf" },
  });
  const candleSeries = chart.addCandlestickSeries({
    upColor: "#d94b63",
    downColor: "#1f9a72",
    borderVisible: false,
    wickUpColor: "#d94b63",
    wickDownColor: "#1f9a72",
  });
  candleSeries.setData(candles);
  chart.addLineSeries({ color: "#2b6cb0", lineWidth: 1, title: "MA20" }).setData(JSON.parse(raw.ma20));
  window.stockChart = { chart, length: candles.length };
  setChartRange(90);
  const resize = () => chart.resize(container.clientWidth, measureChartHeight(container));
  if (window.ResizeObserver) new ResizeObserver(resize).observe(container);
  window.addEventListener("resize", resize);
}

document.addEventListener("click", (event) => {
  const reportRetry = event.target.closest("[data-report-retry]");
  if (reportRetry) {
    window.location.reload();
    return;
  }

  const reportFilter = event.target.closest("[data-report-filter]");
  if (reportFilter) {
    const controls = reportFilter.closest("[data-report-filters]");
    const selectedType = reportFilter.dataset.reportFilter;
    controls.querySelectorAll("[data-report-filter]").forEach((item) => {
      const active = item === reportFilter;
      item.classList.toggle("active", active);
      item.setAttribute("aria-pressed", String(active));
    });
    document.querySelectorAll("[data-report-type]").forEach((lane) => {
      lane.hidden = selectedType !== "all" && lane.dataset.reportType !== selectedType;
    });
    return;
  }

  const watchlist = event.target.closest("[data-watchlist-toggle]");
  if (watchlist) {
    toggleWatchlist(watchlist);
    return;
  }

  const logout = event.target.closest("[data-account-logout]");
  if (logout && window.absorbAccount) {
    fetch("/auth/logout", {
      method: "POST",
      headers: { "X-CSRF-Token": window.absorbAccount.csrf_token },
    }).then((response) => {
      if (response.ok || response.redirected) window.location.assign("/");
    });
    return;
  }

  const preset = event.target.closest("[data-amount-preset]");
  if (preset) {
    const input = bySelector("[data-investment-amount]", preset.closest("[data-return-calculator]"));
    if (input) {
      input.value = preset.dataset.amountPreset;
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }
  }

  const filter = event.target.closest("[data-news-filter]");
  if (filter) {
    const panel = filter.closest(".news-panel");
    const entries = panel.querySelectorAll("[data-news-direction]");
    if (!entries.length) return;
    const direction = filter.dataset.newsFilter;
    let visible = 0;
    panel.querySelectorAll("[data-news-filter]").forEach((item) => {
      const active = item === filter;
      item.classList.toggle("active", active);
      item.setAttribute("aria-pressed", active);
    });
    entries.forEach((item) => {
      item.hidden = direction !== "all" && item.dataset.newsDirection !== direction;
      if (!item.hidden) visible += 1;
    });
    const empty = bySelector("[data-news-filter-empty]", panel);
    if (empty) empty.hidden = visible > 0;
  }

  const range = event.target.closest("[data-chart-range]");
  if (!range) return;
  document.querySelectorAll("[data-chart-range]").forEach((item) => {
    item.classList.toggle("active", item === range);
    item.setAttribute("aria-pressed", item === range);
  });
  setChartRange(Number(range.dataset.chartRange));
});

migrateLegacyHashRoute();
loadDashboard();
loadAccountState();
initStockChart();
initReturnCalculator();
initConversations();
initQuickAsk();
