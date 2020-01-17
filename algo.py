import logbook
import numpy as np
import statsmodels.api as sm
from pipeline_live.data.alpaca.pricing import USEquityPricing
from pylivetrader.algorithm import *
from pylivetrader.api import *
from zipline.pipeline import Pipeline
from zipline.pipeline.factors import ExponentialWeightedMovingAverage, RSI, SimpleMovingAverage, CustomFactor

log = logbook.Logger('algo')


def initialize(context):
    # TODO: Merge all get_prices and enter_postions functions. Run every Hour till 12
    #### Moring ####
    # Close all positions => Get Long/Short positions => Enter Positions => Cancel unfilled orders
    schedule_function(func=close_positions, date_rule=date_rules.every_day(), time_rule=time_rules.market_open())
    schedule_function(func=get_prices, date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open(minutes=6))
    schedule_function(func=enter_positions, date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open(minutes=9))
    schedule_function(func=cancel_open_orders, date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open(minutes=15))

    #### Once an Hour ####
    # Get Long/Short positions => Enter Positions
    for i in range(1, 3):
        schedule_function(func=get_prices_midday, date_rule=date_rules.every_day(),
                          time_rule=time_rules.market_open(hours=i, minutes=6))
        schedule_function(func=enter_positions_midday, date_rule=date_rules.every_day(),
                          time_rule=time_rules.market_open(hours=i, minutes=9))

    # Close all positions and Record Vars
    schedule_function(func=close_positions, date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(minutes=15))


    for i in range(25, 185, 10):  # (low, high, every i minutes)
        # take profits/losses every hour
        schedule_function(func=take_profits, date_rule=date_rules.every_day(),
                          time_rule=time_rules.market_open(minutes=i))

    for i in range(189, 369, 5):  # (low, high, every i minutes)
        # take profits/losses every hour
        schedule_function(func=take_profits, date_rule=date_rules.every_day(),
                          time_rule=time_rules.market_open(minutes=i))

    context.longs = []
    context.shorts = []

    # If we take a loss on trade, dont trade that security for X days
    context.waits = {}
    context.waits_max = 5  # trading days
    context.max_lev = 0.2

    context.spy = symbol('SPY')

    # Hold Setups
    context.holds = {}
    context.hold_max = 3

    # Take profit setup
    context.profit_threshold = 0.06
    context.loss_threshold = -0.01
    context.profit_logging = True

    # market based long/short percentages
    context.max_leverage = 1
    context.long_cnt = 8
    context.short_cnt = 8
    context.max_conc = 0.3

    # Create our pipeline and attach it to our algorithm.
    my_pipe = make_pipeline()
    attach_pipeline(my_pipe, 'my_pipeline')


class High(CustomFactor):
    window_length = 5
    inputs = [USEquityPricing.close]

    def compute(self, today, assets, out, close_prices):
        out[:] = np.max(close_prices, axis=0)


class Low(CustomFactor):
    window_length = 5
    inputs = [USEquityPricing.close]

    def compute(self, today, assets, out, close_prices):
        out[:] = np.min(close_prices, axis=0)


class AveDayRangePerc(CustomFactor):
    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]

    def compute(self, today, assets, out, high, low, close):
        range_percent = (high - low) / close
        out[:] = np.nanmean(range_percent, axis=0)


def make_pipeline():
    """
    Create our pipeline.
    """
    cur_price = USEquityPricing.close.latest
    vol = SimpleMovingAverage(
        inputs=[USEquityPricing.volume],
        window_length=15,
    )
    rng = AveDayRangePerc(
        window_length=25
    )
    ewma5 = ExponentialWeightedMovingAverage.from_span(
        inputs=[USEquityPricing.close],
        window_length=5,
        span=2.5
    )
    rsi = RSI(
        inputs=[USEquityPricing.close],
        window_length=3
    )
    high = High()
    low = Low()
    dol_vol = cur_price * vol
    universe = (
            (dol_vol > 5000000)
            & (rng > 0.02)
            & ((rsi > 70) | (rsi < 30))
            & ewma5.notnan() & ewma5.notnull()
            & rsi.notnan() & rsi.notnull()
            & high.notnan() & high.notnull()
            & low.notnan() & low.notnull()
            & vol.notnan() & vol.notnull()
            & cur_price.notnan() & cur_price.notnull()
    )

    return Pipeline(
        columns={
            "ewma5": ewma5,
            "rsi": rsi,
            "high": high,
            "low": low,
            "vol": vol
        },
        screen=universe
    )


def before_trading_start(context, data):
    # Gets our pipeline output every day.
    context.output = pipeline_output('my_pipeline')


def get_prices(context, data):
    log.info("Getting Positions")

    # Clear Shorts and Longs
    context.shorts = []
    context.longs = []

    # Remove Waits from Output
    Universe500 = context.output.index.tolist()
    if len(Universe500) == 0:
        return
    # Get today's pricing and volume
    intraday_price = data.current(Universe500, 'price')
    vol_data = data.current(Universe500, 'volume')

    rvol_df = vol_data * 100
    rvol_df.columns = ["rvol"]

    today_price_df = intraday_price
    today_price_df.columns = ["cur_price"]

    # Joins intraday dfs to Output
    context.output = context.output.join(today_price_df, how='outer')
    context.output = context.output.join(rvol_df, how='outer')

    # Create percent off 5 day ewma metric
    context.output["per_off_ewma"] = (context.output["price"] - context.output["ewma5"]) / context.output["ewma5"]

    # Add Shorts and Longs to Context
    context.shorts.extend(context.output.query(
        "rsi > 70 and price < low and volume > vol"
    ).nsmallest(context.short_cnt, "per_off_ewma").index.tolist())
    context.longs.extend(context.output.query(
        "rsi < 30 and price > high and volume > vol"
    ).nlargest(context.long_cnt, "per_off_ewma").index.tolist())
    log.info("Longs: {}".format(context.longs))
    log.info("Shorts: {}".format(context.shorts))


def enter_positions(context, data):
    """
    Rebalance daily.
    """
    log.info("Entering Positions")
    if len(context.longs) > 0:
        history = data.history(context.longs, 'close', 5, '1m').bfill().ffill()
        for i, s in enumerate(context.longs):
            if slope(history[s]) < 0 or (history[s].iloc[-1] - history[s].iloc[0]) < 0:
                log.info("Skipping LONG entry {}".format(s))
                del context.longs[i]
    if len(context.shorts) > 0:
        history = data.history(context.shorts, 'close', 5, '1m').bfill().ffill()
        for i, s in enumerate(context.shorts):
            if slope(history[s]) > 0 or (history[s].iloc[-1] - history[s].iloc[0]) > 0:
                log.info("Skipping SHORT entry {}".format(s))
                del context.shorts[i]
    total_positions = len(context.shorts) + len(context.longs)
    pos_size = 0
    if total_positions > 0:
        pos_size = min((context.max_leverage / total_positions), context.max_conc)
    for security in context.longs:
        if data.can_trade(security):
            price = data.current(security, 'price')
            order_target_percent(
                security,
                pos_size,
                limit_price=(price * 1.0025)
            )
    for security in context.shorts:
        if data.can_trade(security):
            price = data.current(security, 'price')
            order_target_percent(
                security,
                (-1 * pos_size),
                limit_price=(price * 0.9975)
            )


def get_prices_midday(context, data):
    log.info("Getting Positions")
    context.shorts = []
    context.longs = []

    # Remove Waits from Output
    if 'price' in context.output.columns and 'volume' in context.output.columns:
        context.output = context.output.drop(['price', 'volume'], 1)
    Universe500 = context.output.index.tolist()
    intraday_price = data.current(Universe500, 'price')

    vol_data = data.current(Universe500, 'volume')
    rvol_df = vol_data * 100

    today_price_df = intraday_price

    # Joins
    context.output = context.output.join(today_price_df, how='outer')
    context.output = context.output.join(rvol_df, how='outer')

    context.output["per_off_ewma"] = (context.output["price"] - context.output["ewma5"]) / context.output["ewma5"]


    context.shorts.extend(context.output.query(
        "rsi > 70 and price < low and volume > vol"
    ).nsmallest(context.short_cnt, "per_off_ewma").index.tolist())
    context.longs.extend(context.output.query(
        "rsi < 30 and price > high and volume > vol"
    ).nlargest(context.long_cnt, "per_off_ewma").index.tolist())
    removals = list(context.portfolio.positions) + list(context.waits.keys())
    context.shorts = list(set(context.shorts) - set(removals))
    context.longs = list(set(context.longs) - set(removals))
    log.info("Longs: {}".format(context.longs))
    log.info("Shorts: {}".format(context.shorts))


def enter_positions_midday(context, data):
    """
    Take profits from existing
    """
    take_profits(context, data)
    avalible_lev = 1 - get_current_leverage(context, data)
    if avalible_lev <= 0:
        return
    if len(context.longs) > 0:
        history = data.history(context.longs, 'close', 9, '1m').bfill().ffill()
        for i, s in enumerate(context.longs):
            if slope(history[s]) < 0 or (history[s].iloc[-1] - history[s].iloc[0]) < 0:
                log.info("Skipping LONG entry {}".format(s))
                del context.longs[i]
    if len(context.shorts) > 0:
        history = data.history(context.shorts, 'close', 9, '1m').bfill().ffill()
        for i, s in enumerate(context.shorts):
            if slope(history[s]) > 0 or (history[s].iloc[-1] - history[s].iloc[0]) > 0:
                log.info("Skipping SHORT entry {}".format(s))
                del context.shorts[i]
    total_positions = len(context.shorts) + len(context.longs)
    pos_size = 0
    if total_positions > 0:
        pos_size = min((avalible_lev / total_positions), context.max_conc)
    for security in context.longs:
        if data.can_trade(security):
            price = data.current(security, 'price')
            order_target_percent(
                security,
                pos_size,
                limit_price=(price * 1.0025)
            )
    for security in context.shorts:
        if data.can_trade(security):
            price = data.current(security, 'price')
            order_target_percent(
                security,
                (-1 * pos_size),
                limit_price=(price * 0.9975)
            )


def take_profits(context, data):
    log.info("Taking Profits")
    cancel_open_orders(context, data)
    positions = context.portfolio.positions
    if len(positions) == 0:
        log.info("No Open Positions")
        return
    history = data.history(list(positions), 'close', 10, '1m').bfill().ffill()
    total_profit = 0
    # if position isn't trending in the right direction and meets Profit/Loss criteria, DUMP
    total_cash = 0
    for s in positions:
        if not data.can_trade(s): continue
        price = data.current(s, 'price')
        amount = positions[s].amount
        profit = (amount / abs(amount)) * ((price / positions[s].cost_basis) - 1)
        if positions[s].amount > 0:
            if slope(history[s]) > 0: continue
            if slope(history[s][-5:]) > 0: continue
            if profit > context.profit_threshold or profit < context.loss_threshold:
                order_target(s, 0, limit_price=price * 0.998)
                wait(context, s, 1)  # start wait
                total_cash += abs(price * amount)
                pnl = (amount * (price - positions[s].cost_basis))
                total_profit += pnl
                if pnl < 0:
                    wait(context, s, 1)
                if context.profit_logging:
                    log.info("Closing Long {} for Profit of ${}".format(s, pnl))
        elif positions[s].amount < 0:
            if slope(history[s]) < 0: continue
            if slope(history[s][-5:]) < 0: continue
            if profit > context.profit_threshold or profit < context.loss_threshold:
                order_target(s, 0, limit_price=price * 1.002)
                wait(context, s, 1)  # start wait
                total_cash += abs(price * amount)
                pnl = (abs(amount) * (positions[s].cost_basis - price))
                total_profit += pnl
                if pnl < 0:
                    wait(context, s, 1)
                if context.profit_logging:
                    log.info("Closing Short {} for Profit of ${}".format(s, pnl))
    if total_cash != 0:
        add_to_winners(context, total_cash, data)


def add_to_winners(context, cash, data):
    winners = {}
    positions = context.portfolio.positions
    history = data.history(list(positions), 'close', 10, '1m').bfill().ffill()
    for s in positions:
        amount = positions[s].amount
        price = data.current(s, 'price')
        profit = (amount / abs(amount)) * ((price / positions[s].cost_basis) - 1)
        if profit > 0.02:
            if amount > 0:
                if slope(history[s]) > 0:
                    winners[s] = 0
            else:
                if slope(history[s]) < 0:
                    winners[s] = 1
    if len(winners) > 0:
        add_amt = cash / max(len(winners), 3)
        for w, l in winners.items():
            price = data.current(w, 'price')
            if l == 0:
                order_value(w, add_amt, limit_price=price * 1.002)
                log.info("adding {} to {}".format(add_amt, w))
            else:
                order_value(w, (-1 * add_amt), limit_price=price * 0.998)
                log.info("adding {} to {}".format((-1 * add_amt), w))


def cancel_open_orders(context, data):
    """Cancel all open orders."""
    for asset, orders in get_open_orders().items():
        for order in orders:
            cancel_order(order)
            log.info("Canceling Order {} for {}".format(order.id, asset))


def close_positions(context, data):
    cancel_open_orders(context, data)
    positions = context.portfolio.positions
    for s in positions:
        order_target_percent(s, 0)
    wait(context)


######## UTILS #########

def get_current_leverage(context, data, consider_open_orders=True):
    long_exposure, short_exposure = get_long_short_exposure(context=context)
    open_order_long_exposure, open_order_short_exposure = get_open_order_long_short_exposure(context=context, data=data)
    curr_cash = context.portfolio.cash - (short_exposure * 2)
    if consider_open_orders:
        curr_cash -= open_order_short_exposure
        curr_cash -= open_order_long_exposure
    curr_leverage = (context.portfolio.portfolio_value - curr_cash) / context.portfolio.portfolio_value
    return curr_leverage


def get_long_short_exposure(context):
    short_exposure = 0.0
    long_exposure = 0.0
    for stock, position in context.portfolio.positions.items():
        amount = position.amount
        last_sale_price = position.last_sale_price
        if amount < 0:
            short_exposure += (last_sale_price * -amount)
        elif amount > 0:
            long_exposure += (last_sale_price * amount)
    return long_exposure, short_exposure


def get_open_order_long_short_exposure(context, data):
    open_order_short_exposure = 0.0
    open_order_long_exposure = 0.0
    for stock, orders in get_open_orders().items():
        price = data.current(stock, 'price')
        if np.isnan(price):
            continue
        amount = 0 if stock not in context.portfolio.positions else context.portfolio.positions[stock].amount
        for oo in orders:
            order_amount = oo.amount - oo.filled
            if order_amount < 0 and amount <= 0:
                open_order_short_exposure += (price * -order_amount)
            elif order_amount > 0 and amount >= 0:
                open_order_long_exposure += (price * order_amount)
    return open_order_long_exposure, open_order_short_exposure


def slope(in_list):  # Return slope of regression line. [Make sure this list contains no nans]
    return sm.OLS(in_list, sm.add_constant(range(-len(in_list) + 1, 1))).fit().params[-1]  # slope


def wait(c, sec=None, action=None):
    if sec and action:
        if action == 1:
            c.waits[sec] = 1  # start wait
        elif action == 0:
            del c.waits[sec]  # end wait
    else:
        for sec in c.waits.copy():
            if c.waits[sec] > c.waits_max:
                del c.waits[sec]
            else:
                c.waits[sec] += 1  # increment
