from pylivetrader.algorithm import *
from pylivetrader.api import *

class ExposureMngr(object):
    """
    Keep track of leverage and long/short exposure

    One Class to rule them all, One Class to define them,
    One Class to monitor them all and in the bytecode bind them

    Usage:
    Define your targets at initialization: I want leverage 1.3  and 60%/40% Long/Short balance
       context.exposure = ExposureMngr(target_leverage = 1.3,
                                       target_long_exposure_perc = 0.60,
                                       target_short_exposure_perc = 0.40)

    update internal state (open orders and positions)
      context.exposure.update(context, data)

    After update is called, you can access the following information:

    how much cash available for trading
      context.exposure.get_available_cash(consider_open_orders = True)
    get long and short available cash as two distinct values
      context.exposure.get_available_cash_long_short(consider_open_orders = True)

    same as account.leverage but this keeps track of open orders
      context.exposure.get_current_leverage(consider_open_orders = True)

    sum of long and short positions current value
      context.exposure.get_exposure(consider_open_orders = True)
    get long and short position values as two distinct values
      context.exposure.get_long_short_exposure(consider_open_orders = True)
    get long and short exposure as percentage
      context.exposure.get_long_short_exposure_pct(consider_open_orders = True,  consider_unused_cash = True)
    """

    def __init__(self, target_leverage=1.0, target_long_exposure_perc=0.50, target_short_exposure_perc=0.50):
        self.target_leverage = target_leverage
        self.target_long_exposure_perc = target_long_exposure_perc
        self.target_short_exposure_perc = target_short_exposure_perc
        self.short_exposure = 0.0
        self.long_exposure = 0.0
        self.open_order_short_exposure = 0.0
        self.open_order_long_exposure = 0.0

    def get_current_leverage(self, context, consider_open_orders=True):
        curr_cash = context.portfolio.cash - (self.short_exposure * 2)
        if consider_open_orders:
            curr_cash -= self.open_order_short_exposure
            curr_cash -= self.open_order_long_exposure
        curr_leverage = (context.portfolio.portfolio_value - curr_cash) / context.portfolio.portfolio_value
        return curr_leverage

    def get_exposure(self, context, consider_open_orders=True):
        long_exposure, short_exposure = self.get_long_short_exposure(context, consider_open_orders)
        return long_exposure + short_exposure

    def get_long_short_exposure(self, context, consider_open_orders=True):
        long_exposure = self.long_exposure
        short_exposure = self.short_exposure
        if consider_open_orders:
            long_exposure += self.open_order_long_exposure
            short_exposure += self.open_order_short_exposure
        return (long_exposure, short_exposure)

    def get_long_short_exposure_pct(self, context, consider_open_orders=True, consider_unused_cash=True):
        long_exposure, short_exposure = self.get_long_short_exposure(context, consider_open_orders)
        total_cash = long_exposure + short_exposure
        if consider_unused_cash:
            total_cash += self.get_available_cash(context, consider_open_orders)
        long_exposure_pct = long_exposure / total_cash if total_cash > 0 else 0
        short_exposure_pct = short_exposure / total_cash if total_cash > 0 else 0
        return (long_exposure_pct, short_exposure_pct)

    def get_available_cash(self, context, consider_open_orders=True):
        curr_cash = context.portfolio.cash - (self.short_exposure * 2)
        if consider_open_orders:
            curr_cash -= self.open_order_short_exposure
            curr_cash -= self.open_order_long_exposure
        leverage_cash = context.portfolio.portfolio_value * (self.target_leverage - 1.0)
        return curr_cash + leverage_cash

    def get_available_cash_long_short(self, context, consider_open_orders=True):
        total_available_cash = self.get_available_cash(context, consider_open_orders)
        long_exposure = self.long_exposure
        short_exposure = self.short_exposure
        if consider_open_orders:
            long_exposure += self.open_order_long_exposure
            short_exposure += self.open_order_short_exposure
        current_exposure = long_exposure + short_exposure + total_available_cash
        target_long_exposure = current_exposure * self.target_long_exposure_perc
        target_short_exposure = current_exposure * self.target_short_exposure_perc
        long_available_cash = target_long_exposure - long_exposure
        short_available_cash = target_short_exposure - short_exposure
        return (long_available_cash, short_available_cash)

    def update(self, context, data):
        #
        # calculate cash needed to complete open orders
        #
        self.open_order_short_exposure = 0.0
        self.open_order_long_exposure = 0.0
        for stock, orders in get_open_orders().items():
            price = data.current(stock, 'price')
            if np.isnan(price):
                continue
            amount = 0 if stock not in context.portfolio.positions else context.portfolio.positions[stock].amount
            for oo in orders:
                order_amount = oo.amount - oo.filled
                if order_amount < 0 and amount <= 0:
                    self.open_order_short_exposure += (price * -order_amount)
                elif order_amount > 0 and amount >= 0:
                    self.open_order_long_exposure += (price * order_amount)

        #
        # calculate long/short positions exposure
        #
        self.short_exposure = 0.0
        self.long_exposure = 0.0
        for stock, position in context.portfolio.positions.items():
            amount = position.amount
            last_sale_price = position.last_sale_price
            if amount < 0:
                self.short_exposure += (last_sale_price * -amount)
            elif amount > 0:
                self.long_exposure += (last_sale_price * amount)