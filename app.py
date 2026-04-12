    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'].squeeze(), high=df['High'].squeeze(),
        low=df['Low'].squeeze(), close=df['Close'].squeeze(),
        name='OHLC',
        increasing=dict(line=dict(color=inc_line, width=1), fillcolor=inc_fill),
        decreasing=dict(line=dict(color=dec_line, width=1), fillcolor=dec_fill),
        whiskerwidth=0.5,
        hovertemplate=(
            "<span style='display:block; text-align:center;'><b>%{x|%b %d, %Y}</b></span>"
            "<b>Open :</b> $%{open:.2f}<br>"
            "<b>High :</b> $%{high:.2f}<br>"
            "<b>Low :</b> $%{low:.2f}<br>"
            "<b>Close :</b> $%{close:.2f}"
            "<extra></extra>"
        )
    ), row=1, col=1)
