    hover_texts = []
    for i in range(len(df)):
        date_str = df.index[i].strftime('%b %d, %Y')
        o = open_series.iloc[i]
        h = high_series.iloc[i]
        l = low_series.iloc[i]
        c = close_series.iloc[i]
        m20 = ma20.iloc[i]
        m50 = ma50.iloc[i]

        is_bullish = c >= o
        triangle = "▲" if is_bullish else "▼"
        triangle_color = "#4ade80" if (is_bullish and IS_DARK) else ("#16a34a" if is_bullish else ("#f87171" if IS_DARK else "#dc2626"))

        txt = (
            f"<b style='display:block; text-align:center;'>"
            f"<span style='color:{triangle_color};'>{triangle}</span>"
            f"&nbsp;{date_str}&nbsp;"
            f"<span style='color:{triangle_color};'>{triangle}</span>"
            f"</b>"
            f"<br>Open  : ${o:.2f}"
            f"<br>High  : ${h:.2f}"
            f"<br>Low   : ${l:.2f}"
            f"<br>Close : ${c:.2f}"
        )
        if not np.isnan(m20):
            txt += f"<br>MA 20 : ${m20:.2f}"
        if not np.isnan(m50):
            txt += f"<br>MA 50 : ${m50:.2f}"

        hover_texts.append(txt)
