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
    #### Moring ####
    # Close all positions => Get Long/Short positions => Enter Positions => Cancel unfilled orders
    schedule_function(func=close_positions, date_rule=date_rules.every_day(), time_rule=time_rules.market_open())
    schedule_function(func=cancel_open_orders, date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open(minutes=25))

    #### Once an Hour ####
    # Get Long/Short positions => Enter Positions
    for i in range(0, 3):
        schedule_function(func=get_prices, date_rule=date_rules.every_day(),
                          time_rule=time_rules.market_open(hours=i, minutes=7))
        schedule_function(func=enter_positions, date_rule=date_rules.every_day(),
                          time_rule=time_rules.market_open(hours=i, minutes=9))

    schedule_function(func=add_available_lev_to_winners, date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open(hours=2, minutes=46))

    # Close all positions and Record Vars
    schedule_function(func=close_positions, date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(minutes=30))

    for i in range(36, 66, 10):  # (low, high, every i minutes)
        # take profits/losses every hour
        schedule_function(func=retry_skipped, date_rule=date_rules.every_day(),
                          time_rule=time_rules.market_open(minutes=i))

    for i in range(76, 176, 10):  # (low, high, every i minutes)
        # take profits/losses every hour
        schedule_function(func=retry_skipped, date_rule=date_rules.every_day(),
                          time_rule=time_rules.market_open(minutes=i))

    for i in range(196, 236, 10):  # (low, high, every i minutes)
        # take profits/losses every hour
        schedule_function(func=retry_skipped, date_rule=date_rules.every_day(),
                          time_rule=time_rules.market_open(minutes=i))

    for i in range(256, 369, 5):  # (low, high, every i minutes)
        # take profits/losses every hour
        schedule_function(func=retry_skipped, date_rule=date_rules.every_day(),
                          time_rule=time_rules.market_open(minutes=i))

    context.longs = []
    context.shorts = []
    context.universe = []
    context.long_misses = []
    context.short_misses = []

    # If we take a loss on trade, dont trade that security for X days
    context.waits = {}
    context.waits_max = 5  # trading days

    context.spy = symbol('SPY')

    # Hold Setups
    context.holds = {}
    context.hold_max = 3
    context.today_entries = {}

    # Take profit setup
    context.profit_threshold = 0.10
    context.loss_threshold = -0.025
    context.profit_logging = True

    # market based long/short percentages
    context.max_leverage = 1.0
    context.long_cnt = 6
    context.short_cnt = 6
    context.max_conc = 0.33

    # Create our pipeline and attach it to our algorithm.
    my_pipe = make_pipeline()
    attach_pipeline(my_pipe, 'my_pipeline')


class High(CustomFactor):
    window_length = 10
    inputs = [USEquityPricing.close]

    def compute(self, today, assets, out, close_prices):
        out[:] = np.max(close_prices, axis=0)


class Low(CustomFactor):
    window_length = 10
    inputs = [USEquityPricing.close]

    def compute(self, today, assets, out, close_prices):
        out[:] = np.min(close_prices, axis=0)


class AveDayRangePerc(CustomFactor):
    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]

    def compute(self, today, assets, out, high, low, close):
        range_percent = (high - low) / close
        out[:] = np.nanmean(range_percent, axis=0)


class Velocity(CustomFactor):
    inputs = [USEquityPricing.close]

    def compute(self, today, asset_ids, out, close_prices):
        out[:] = (close_prices[-1] - np.nanmean(close_prices)) / close_prices[-1]


def make_pipeline():
    """
    Create our pipeline.
    """
    cur_price = USEquityPricing.close.latest
    rng = AveDayRangePerc(
        window_length=15
    )
    vol = SimpleMovingAverage(
        inputs=[USEquityPricing.volume],
        window_length=15
    )
    dollar_vol = cur_price * vol
    mask = (cur_price > 1) & (rng > 0.02) & (dollar_vol > 5000000)
    ewma5 = ExponentialWeightedMovingAverage.from_span(
        inputs=[USEquityPricing.close],
        window_length=5,
        span=2.5,
        mask=mask
    )
    rsi = RSI(
        inputs=[USEquityPricing.close],
        window_length=3,
        mask=mask
    )
    high = High(mask=mask)
    low = Low(mask=mask)
    velo = Velocity(window_length=50, mask=mask)
    universe = (((velo > 0) & (rsi < 50)) | ((velo < 0) & (rsi > 50))) & mask
    return Pipeline(
        columns={
            "ewma5": ewma5,
            "rsi": rsi,
            "high": high,
            "low": low,
            "vol": vol,
            "velo": velo
        },
        screen=universe
    )


def before_trading_start(context, data):
    # Gets our pipeline output every day.
    context.today_entries = {}
    context.output = pipeline_output('my_pipeline')
    context.universe = context.output.index.tolist()
    context.long_cnt = 6
    context.short_cnt = 6
    context.long_misses = []
    context.short_misses = []


def get_prices(context, data):
    log.info("Getting Positions")
    if pd.Series(['price', 'volume', 'per_off_ewma']).isin(context.output.columns).all():
        context.output = context.output.drop(['price', 'volume', 'per_off_ewma'], 1)

    # Remove Waits from Output

    if len(context.universe) == 0:
        return
    # Get today's pricing and volume
    intraday_price = data.current(context.universe, 'price')
    vol_data = data.current(context.universe, 'volume')

    rvol_df = vol_data * 100
    today_price_df = intraday_price

    # Joins intraday dfs to Output
    context.output = context.output.join(today_price_df, how='outer')
    context.output = context.output.join(rvol_df, how='outer')

    # Create percent off 5 day ewma metric
    context.output["per_off_ewma"] = (context.output["price"] - context.output["ewma5"]) / context.output["ewma5"]

    context.shorts = []
    context.longs = []
    if get_datetime().hour < 15:
        log.info(get_datetime().hour)
        short_query = "rsi > 50 and price < low and volume > vol and velo < 0 and per_off_ewma < -0.05"
        long_query = "rsi < 50 and price > high and volume > vol and velo > 0 and per_off_ewma > 0.04"
        remove_list = list(context.waits.keys())
    else:
        remove_list = list(context.portfolio.positions) + list(context.waits.keys()) + list(
            context.today_entries.keys())
        short_query = "rsi > 50 and price < low and volume > vol and velo < 0 and per_off_ewma < -0.05"
        long_query = "rsi < 50 and price > high and volume > vol and velo > 0 and per_off_ewma > 0.04"

    # Add Shorts and Longs to Context
    context.shorts.extend(context.output.query(
        short_query
    ).nsmallest(context.short_cnt, "per_off_ewma").index.tolist())
    context.longs.extend(context.output.query(
        long_query
    ).nlargest(context.long_cnt, "per_off_ewma").index.tolist())
    context.shorts = list(set(context.shorts) - set(remove_list))
    context.longs = list(set(context.longs) - set(remove_list))
    log.info("Longs: {}".format(context.longs))
    log.info("Shorts: {}".format(context.shorts))


def enter_positions(context, data):
    """
    Rebalance daily.
    """
    log.info("Entering Positions")
    if get_datetime().hour < 15:
        available_lev = context.max_leverage
    else:
        take_profits(context, data)
        available_lev = context.max_leverage - get_current_leverage(context, data)
    if available_lev <= 0.1:
        return
    if len(context.longs) > 0:
        history = data.history(context.longs, 'close', 7, '1m').bfill().ffill()
        for i, s in enumerate(context.longs):
            if slope(history[s]) < 0 or (history[s].iloc[-1] - history[s].iloc[0]) < 0:
                del context.longs[i]
                add_to_misses(context, s, False)
                log.info("Skipping Long Entry {}".format(s))
    if len(context.shorts) > 0:
        history = data.history(context.shorts, 'close', 7, '1m').bfill().ffill()
        for i, s in enumerate(context.shorts):
            if slope(history[s]) > 0 or (history[s].iloc[-1] - history[s].iloc[0]) > 0:
                del context.shorts[i]
                add_to_misses(context, s, True)
                log.info("Skipping Short Entry {}".format(s))
    total_positions = len(context.shorts) + len(context.longs)
    if total_positions > 0:
        pos_size = min((available_lev / total_positions), context.max_conc)
    else:
        if len(context.long_misses) + len(context.short_misses) == 0:
            add_available_lev_to_winners(context, data)
        return
    for security in context.longs:
        if data.can_trade(security):
            price = data.current(security, 'price')
            order_target_percent(
                security,
                pos_size,
                stop_price=(price * 1.0025)
            )
            context.today_entries[security] = price
    for security in context.shorts:
        if data.can_trade(security):
            price = data.current(security, 'price')
            order_target_percent(
                security,
                (-1 * pos_size),
                stop_price=(price * 0.9975)
            )
            context.today_entries[security] = price


def take_profits(context, data):
    log.info("Taking Profits")
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


def retry_skipped(context, data):
    log.info("Starting Retry Skipped")
    if len(context.long_misses) + len(context.short_misses) == 0:
        add_available_lev_to_winners(context, data)
        return
    cancel_open_orders(context, data)
    available_lev = 1 - get_current_leverage(context, data)
    if available_lev <= 0.1:
        take_profits(context, data)
        return
    long_enters = []
    short_enters = []
    if len(context.long_misses) > 0:
        history = data.history(context.long_misses, 'close', 9, '1m').bfill().ffill()
        for i, s in enumerate(context.long_misses):
            if slope(history[s]) > 0 and (history[s].iloc[-1] - history[s].iloc[0]) > 0:
                del context.long_misses[i]
                long_enters.append(s)
            else:
                log.info("Skipping Long Entry {}".format(s))
    if len(context.short_misses) > 0:
        history = data.history(context.short_misses, 'close', 9, '1m').bfill().ffill()
        for i, s in enumerate(context.short_misses):
            if slope(history[s]) < 0 and (history[s].iloc[-1] - history[s].iloc[0]) < 0:
                del context.short_misses[i]
                short_enters.append(s)
            else:
                log.info("Skipping Short Entry {}".format(s))
    total_positions = len(short_enters) + len(long_enters)
    if total_positions > 0:
        pos_size = min((available_lev / total_positions), context.max_conc)
    else:
        pos_size = 0
    for security in long_enters:
        if data.can_trade(security):
            price = data.current(security, 'price')
            order_target_percent(
                security,
                pos_size,
                stop_price=(price * 1.0025)
            )
            context.today_entries[security] = price
    for security in short_enters:
        if data.can_trade(security):
            price = data.current(security, 'price')
            order_target_percent(
                security,
                (-1 * pos_size),
                stop_price=(price * 0.9975)
            )
            context.today_entries[security] = price
    take_profits(context, data)


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
                order_value(w, add_amt, stop_price=price * 1.002)
                log.info("adding {} to {}".format(add_amt, w))
            else:
                order_value(w, (-1 * add_amt), stop_price=price * 0.998)
                log.info("adding {} to {}".format((-1 * add_amt), w))


def add_available_lev_to_winners(context, data):
    log.info("Adding available to winners")
    available_lev = 1 - get_current_leverage(context, data)
    positions = context.portfolio.positions
    if available_lev < 0.1 or len(positions) == 0:
        return
    history = data.history(list(positions), 'close', 10, '1m').bfill().ffill()
    winners = {}
    for s in positions:
        amount = positions[s].amount
        price = data.current(s, 'price')
        profit = (amount / abs(amount)) * ((price / positions[s].cost_basis) - 1)
        if profit > 0.01:
            if amount > 0:
                if slope(history[s]) > 0:
                    winners[s] = 0
            else:
                if slope(history[s]) < 0:
                    winners[s] = 1
    if len(winners) > 0:
        pos_size = min((available_lev / len(winners)), context.max_conc)
        for w, l in winners.items():
            if l == 0:
                order_percent(w, pos_size)
                if context.profit_logging:
                    log.info("adding {} to {}".format(pos_size, w))
            else:
                order_percent(w, (-1 * pos_size))
                if context.profit_logging:
                    log.info("adding {} to {}".format((-1 * pos_size), w))


def cancel_open_orders(context, data):
    """Cancel all open orders."""
    for asset, orders in get_open_orders().items():
        for order in orders:
            cancel_order(order)
            if order.amount < 0:
                add_to_misses(context, asset, True)
            else:
                add_to_misses(context, asset, False)
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


def add_to_misses(context, s, short):
    if short:
        if s not in context.short_misses:
            context.short_misses.append(s)
    else:
        if s not in context.long_misses:
            context.long_misses.append(s)
