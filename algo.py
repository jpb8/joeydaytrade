import numpy as np
import pandas as pd
from zipline.pipeline import Pipeline
from zipline.pipeline.factors import ExponentialWeightedMovingAverage, RSI, SimpleMovingAverage, CustomFactor
from pipeline_live.data.alpaca.pricing import USEquityPricing
from pylivetrader.api import *
from pylivetrader.algorithm import *
import statsmodels.api as sm

from .exposure import ExposureMngr

import logbook

log = logbook.Logger('algo')


def initialize(context):
    # TODO: Merge all get_prices and enter_postions functions. Run every Hour till 12
    #### Moring ####
    # Close all positions => Get Long/Short positions => Enter Positions => Cancel unfilled orders
    schedule_function(func=close_positions, date_rule=date_rules.every_day(), time_rule=time_rules.market_open())
    schedule_function(func=get_prices, date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open(minutes=7))
    schedule_function(func=enter_positions, date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open(minutes=10))
    schedule_function(func=cancel_open_orders, date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open(minutes=15))

    #### Once an Hour ####
    # Get Long/Short positions => Enter Positions
    for i in range(1, 5):
        schedule_function(func=get_prices_midday, date_rule=date_rules.every_day(),
                          time_rule=time_rules.market_open(hours=i, minutes=7))
        schedule_function(func=enter_positions_midday, date_rule=date_rules.every_day(),
                          time_rule=time_rules.market_open(hours=i, minutes=10))

    # Close all positions and Record Vars
    schedule_function(func=close_positions, date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(minutes=15))
    schedule_function(func=my_record_vars, date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(minutes=10))

    # Set commissions and slippage to 0 to determine pure alpha
    # set_commission(commission.PerShare(cost=0, min_trade_cost=0))
    # set_slippage(slippage.FixedSlippage(spread=0))

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
    context.profit_threshold = .06
    context.loss_threshold = -.02
    context.profit_logging = True

    # market based long/short percentages
    context.long_leverage = 0.6
    context.long_cnt = 12
    context.short_leverage = -0.4
    context.short_cnt = 8
    context.max_conc = 0.2
    context.exposure = ExposureMngr(target_leverage=1.0,
                                    target_long_exposure_perc=0.50,
                                    target_short_exposure_perc=0.50)

    for i in range(189, 369, 5):  # (low, high, every i minutes)
        # take profits/losses every hour
        schedule_function(func=take_profits, date_rule=date_rules.every_day(),
                          time_rule=time_rules.market_open(minutes=i))

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
    universe = (
            (vol > 250000)
            & (rng > 0.025)
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


def calc_leverage_settings(c, spy_slope, spy_returns):
    # Set Long/Short Security Numbers and Leverage
    # if SPY is down 2% set algo to favor short
    # Change to SPY Momentum?
    if spy_slope >= -0.02 and spy_returns > -0.006:
        log.info("LONG")
        c.long_leverage = 0.6
        c.long_cnt = 12
        c.short_leverage = -0.4
        c.short_cnt = 8
    else:
        log.info("SHORT")
        c.long_leverage = 0.4
        c.long_cnt = 8
        c.short_leverage = -0.6
        c.short_cnt = 12


def get_prices(context, data):
    log.info("Getting Positions")
    spy_price = data.history(assets=[context.spy], fields='price', bar_count=100, frequency='1d')
    spy_slope = slope(spy_price[context.spy])
    spy_current_rets = (spy_price.iloc[-1] - spy_price.iloc[-2]) / spy_price.iloc[-1]

    calc_leverage_settings(context, spy_slope, spy_current_rets[context.spy])

    # Remove Waits from Output
    context.output = context.output.drop(context.output.index[[list(context.waits.values())]])
    Universe500 = context.output.index.tolist()
    if len(Universe500) == 0:
        return
    # Get today's pricing and volume
    intraday_price = data.current(Universe500, 'price')
    vol_data = data.current(Universe500, 'volume')

    rvol_df = vol_data * 255
    rvol_df.columns = ["rvol"]

    today_price_df = intraday_price
    today_price_df.columns = ["cur_price"]

    # Joins intraday dfs to Output
    context.output = context.output.join(today_price_df, how='outer')
    context.output = context.output.join(rvol_df, how='outer')

    # Create percent off 5 day ewma metric
    context.output["per_off_ewma"] = (context.output["price"] - context.output["ewma5"]) / context.output["ewma5"]

    # Clear Shorts and Longs
    context.shorts = []
    context.longs = []

    # Add Shorts and Longs to Context
    context.shorts.extend(context.output.query(
        "rsi > 70 and price < low and volume > vol"
    ).nsmallest(context.short_cnt, "per_off_ewma").index.tolist())
    context.longs.extend(context.output.query(
        "rsi < 30 and price > high and volume > vol"
    ).nlargest(context.long_cnt, "per_off_ewma").index.tolist())
    context.short_cnt = len(context.shorts)
    context.long_cnt = len(context.longs)
    log.info("Longs: {}".format(context.longs))
    log.info("Shorts: {}".format(context.shorts))


def enter_positions(context, data):
    """
    Rebalance daily.
    """
    log.info("Entering Positions")
    long_conc = 0
    short_conc = 0
    if len(context.longs) > 0:
        history = data.history(context.longs, 'close', 9, '1m').bfill().ffill()
        for i, s in enumerate(context.longs):
            if slope(history[s]) < 0:
                del context.longs[i]
    if len(context.longs) > 0:
        long_conc = min(context.long_leverage / (len(context.longs)), context.max_conc)
    for security in context.longs:
        if data.can_trade(security):
            order_target_percent(
                security,
                long_conc
            )
    if len(context.shorts) > 0:
        history = data.history(context.shorts, 'close', 9, '1m').bfill().ffill()
        for i, s in enumerate(context.shorts):
            if slope(history[s]) < 0:
                del context.shorts[i]
    if len(context.shorts) > 0:
        short_conc = max(context.short_leverage / (len(context.shorts)), (-1 * context.max_conc))
    for security in context.shorts:
        if data.can_trade(security):
            order_target_percent(
                security,
                short_conc
            )


def get_prices_midday(context, data):
    spy_price = data.history([sid(8554)], 'price', 100, '1d')
    spy_slope = slope(spy_price[sid(8554)])
    spy_current_rets = (spy_price.iloc[-1] - spy_price.iloc[-2]) / spy_price.iloc[-1]

    calc_leverage_settings(context, spy_slope, spy_current_rets[sid(8554)])

    # Remove Waits from Output
    context.output = context.output.drop(['price', 'volume'], 1)
    Universe500 = context.output.index.tolist()
    intraday_price = data.current(Universe500, 'price')

    vol_data = data.current(Universe500, 'volume')
    rvol_df = vol_data * 225
    rvol_df.columns = ['vol_noon']

    today_price_df = intraday_price
    today_price_df.columns = ['price_noon']

    # Joins
    context.output = context.output.join(today_price_df, how='outer')
    context.output = context.output.join(rvol_df, how='outer')

    context.output["per_off_ewma"] = (context.output["price"] - context.output["ewma5"]) / context.output["ewma5"]

    context.shorts = []
    context.longs = []

    context.short_cnt -= len(context.shorts)
    context.long_cnt -= len(context.longs)

    context.shorts.extend(context.output.query(
        "rsi > 70 and price < low and volume > vol"
    ).nsmallest(context.short_cnt, "per_off_ewma").index.tolist())
    context.longs.extend(context.output.query(
        "rsi < 30 and price > high and volume > vol"
    ).nlargest(context.long_cnt, "per_off_ewma").index.tolist())
    positions = list(context.portfolio.positions)
    context.shorts = list(set(context.shorts) - set(positions))
    context.longs = list(set(context.longs) - set(positions))
    context.short_cnt = len(context.shorts)
    context.long_cnt = len(context.longs)
    log.info("Longs: {}".format(context.longs))
    log.info("Shorts: {}".format(context.shorts))


def enter_positions_midday(context, data):
    """
    Take profits from existing
    """
    take_profits(context, data)
    context.exposure.update(context, data)
    avalible_lev = 1 - context.exposure.get_current_leverage(context)
    aval_long_lev = context.long_leverage * avalible_lev
    aval_short_lev = context.short_leverage * avalible_lev
    if avalible_lev <= 0:
        return
    if len(context.longs) > 0:
        history = data.history(context.longs, 'close', 9, '1m').bfill().ffill()
        for i, s in enumerate(context.longs):
            if slope(history[s]) < 0:
                del context.longs[i]
    if len(context.longs) > 0:
        long_conc = min(aval_long_lev / (len(context.longs)), context.max_conc)
    for security in context.longs:
        if data.can_trade(security):
            order_target_percent(
                security,
                long_conc
            )
    if len(context.shorts) > 0:
        history = data.history(context.shorts, 'close', 9, '1m').bfill().ffill()
        for i, s in enumerate(context.shorts):
            if slope(history[s]) < 0:
                del context.shorts[i]
    if len(context.shorts) > 0:
        short_conc = max(aval_short_lev / (len(context.shorts)), (-1 * context.max_conc))
    for security in context.shorts:
        if data.can_trade(security):
            order_target_percent(
                security,
                short_conc
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


def slope(in_list):  # Return slope of regression line. [Make sure this list contains no nans]
    return sm.OLS(in_list, sm.add_constant(range(-len(in_list) + 1, 1))).fit().params[-1]  # slope


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
            if l == 0:
                order_value(w, add_amt)
                log.info("adding {} to {}".format(add_amt, w))
            else:
                order_value(w, (-1 * add_amt))
                log.info("adding {} to {}".format((-1 * add_amt), w))


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


def cancel_open_orders(context, data):
    """Cancel all open orders."""
    for asset, orders in get_open_orders().items():
        for order in orders:
            cancel_order(order)
            log.info("Canceling Order {} for {}".format(order.id, asset))


def close_positions(context, data):
    cancel_open_orders(context, data)
    record(leverage=context.account.leverage, long_count=len(context.longs), short_count=len(context.shorts))
    positions = context.portfolio.positions
    for s in positions:
        order_target_percent(s, 0)


def my_record_vars(context, data):
    """
    Record variables at the end of each day.
    """
    # Run waits
    wait(context)
    # Record our variables.
    if context.profit_logging:
        log.info("Today's shorts: " + ", ".join([short_.symbol for short_ in context.shorts]))
        log.info("Today's longs: " + ", ".join([long_.symbol for long_ in context.longs]))
