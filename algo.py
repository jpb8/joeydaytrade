import numpy as np
import pandas as pd
from zipline.pipeline import Pipeline
from zipline.pipeline.factors import ExponentialWeightedMovingAverage, RSI, SimpleMovingAverage, CustomFactor
from pipeline_live.data.alpaca.pricing import USEquityPricing
import os
from pylivetrader.api import *
from pylivetrader.algorithm import *

import logbook

log = logbook.Logger('algo')

def initialize(context):
    # Schedule our rebalance function to run at the start of each day.
    schedule_function(func=enter_positions, date_rule=date_rules.every_day(),
                           time_rule=time_rules.market_open(minutes=25))
    schedule_function(func=cancel_open_orders, date_rule=date_rules.every_day(),
                           time_rule=time_rules.market_open(minutes=30))

    # Record variables at the end of each day.
    schedule_function(func=my_record_vars, date_rule=date_rules.every_day(),
                           time_rule=time_rules.market_close(minutes=10))

    schedule_function(func=close_positions, date_rule=date_rules.every_day(),
                           time_rule=time_rules.market_close(minutes=15))
    schedule_function(func=close_positions, date_rule=date_rules.every_day(), time_rule=time_rules.market_open())

    # Get intraday prices and create Short/Long lists
    schedule_function(func=get_prices, date_rule=date_rules.every_day(),
                           time_rule=time_rules.market_open(minutes=20))

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


    for i in range(185, 380, 5):  # (low, high, every i minutes)
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
        # log.info("LONG")
        c.long_leverage = 0.6
        c.long_cnt = 12
        c.short_leverage = -0.4
        c.short_cnt = 8
    else:
        # log.info("SHORT")
        c.long_leverage = 0.4
        c.long_cnt = 8
        c.short_leverage = -0.6
        c.short_cnt = 12


def get_prices(context, data):
    spy_price = data.history([context.spy], 'price', 100, '1d')
    spy_slope = slope(spy_price[context.spy])
    spy_current_rets = (spy_price.iloc[-1] - spy_price.iloc[-2]) / spy_price.iloc[-1]

    calc_leverage_settings(context, spy_slope, spy_current_rets[context.spy])

    # Remove Waits from Output
    context.output = context.output.drop(context.output.index[[list(context.waits.values())]])
    Universe500 = context.output.index.tolist()
    intraday_price = data.history(Universe500, 'close', 6, '1m')
    intraday_ret = (intraday_price.iloc[-1] - intraday_price.iloc[0]) / intraday_price.iloc[0]

    vol_data = data.history(Universe500, 'volume', 6, '1m')
    rvol = vol_data.mean() * 225
    rvol_df = pd.DataFrame(rvol)
    rvol_df.columns = ["rvol"]

    today_price = intraday_price.iloc[-1]
    today_price_df = pd.DataFrame(today_price)
    today_price_df.columns = ["cur_price"]

    # Create Today's price Metric
    intraday_ret_df = pd.DataFrame(intraday_ret)
    intraday_ret_df.columns = ["intraday_return"]

    # Joins
    context.output = context.output.join(today_price_df, how='outer')
    context.output = context.output.join(intraday_ret_df, how='outer')
    context.output = context.output.join(rvol_df, how='outer')

    context.output["per_off_ewma"] = (context.output["ewma5"] - context.output["cur_price"]) / context.output["ewma5"]

    context.shorts = []
    context.longs = []

    context.short_cnt -= len(context.shorts)
    context.long_cnt -= len(context.longs)

    context.shorts.extend(context.output.query(
        "rsi > 70 and cur_price < low and rvol > vol"
    ).nlargest(context.short_cnt, "per_off_ewma").index.tolist())
    context.longs.extend(context.output.query(
        "rsi < 30 and cur_price > high and rvol > vol"
    ).nsmallest(context.long_cnt, "per_off_ewma").index.tolist())
    context.short_cnt = len(context.shorts)
    context.long_cnt = len(context.longs)


def enter_positions(context, data):
    """
    Rebalance daily.
    """
    if len(context.longs) > 0:
        long_conc = min(context.long_leverage / (len(context.longs)), context.max_conc)
    for security in context.longs:
        if data.can_trade(security):
            order_target_percent(
                security,
                long_conc
            )
    if len(context.shorts) > 0:
        short_conc = max(context.short_leverage / (len(context.shorts)), (-1 * context.max_conc))
    for security in context.shorts:
        if data.can_trade(security):
            order_target_percent(
                security,
                short_conc
            )


def take_profits(context, data):
    positions = context.portfolio.positions
    if len(positions) == 0: return
    history = data.history(list(positions), 'close', 10, '1m')
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
            if history[s][-1] > history[s][-2]: continue
            if profit > context.profit_threshold or profit < context.loss_threshold:
                order_target(s, 0)
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
            if history[s][-1] < history[s][-2]: continue
            if profit > context.profit_threshold or profit < context.loss_threshold:
                order_target(s, 0)
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


import statsmodels.api as sm


def slope(in_list):  # Return slope of regression line. [Make sure this list contains no nans]
    return sm.OLS(in_list, sm.add_constant(range(-len(in_list) + 1, 1))).fit().params[-1]  # slope


def add_to_winners(context, cash, data):
    winners = {}
    positions = context.portfolio.positions
    history = data.history(list(positions), 'close', 10, '1m')
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
