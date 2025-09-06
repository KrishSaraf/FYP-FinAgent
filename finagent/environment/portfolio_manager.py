import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, date
from dataclasses import dataclass, field
from enum import Enum
import json
from finagent.registry import ENVIRONMENT

class PositionSide(Enum):
    """Position side enumeration"""
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"

@dataclass
class Position:
    """Individual position in a stock with FIFO lots for accurate gains/holding periods"""
    stock_symbol: str
    side: PositionSide
    lots: List[Dict[str, Any]] = field(default_factory=list)  # each lot: {'quantity', 'price', 'entry_date', 'fees'}
    current_price: float = 0.0
    last_updated: datetime = None

    @property
    def quantity(self) -> int:
        return sum(int(lot['quantity']) for lot in self.lots)

    @property
    def entry_price(self) -> float:
        total_qty = self.quantity
        if total_qty == 0:
            return 0.0
        total_cost = sum(lot['quantity'] * lot['price'] + lot.get('fees', 0.0) for lot in self.lots)
        return total_cost / total_qty

    @property
    def entry_date(self) -> Optional[datetime]:
        # earliest lot entry date
        if not self.lots:
            return None
        return min(lot['entry_date'] for lot in self.lots)

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        # Sum over lots
        pnl = 0.0
        for lot in self.lots:
            pnl += (self.current_price - lot['price']) * lot['quantity']
        return pnl

    @property
    def unrealized_pnl_percent(self) -> float:
        entry_value = sum(lot['price'] * lot['quantity'] + lot.get('fees', 0.0) for lot in self.lots)
        if entry_value == 0:
            return 0.0
        return (self.unrealized_pnl / entry_value) * 100

@dataclass
class Transaction:
    """Trading transaction record"""
    transaction_id: str
    stock_symbol: str
    transaction_type: str  # "BUY", "SELL", "SHORT", "COVER"
    quantity: int  # Changed to int for whole shares
    price: float
    timestamp: datetime
    fees: float = 0.0
    notes: str = ""
    
    @property
    def total_value(self) -> float:
        """Total transaction value including fees"""
        return (self.quantity * self.price) + self.fees

@ENVIRONMENT.register_module()
class PortfolioManager:
    """
    Manages portfolio state for RL trading environment.
    Tracks positions, cash, P&L, and portfolio metrics.
    """
    
    def __init__(self, 
                 initial_cash: float = 1000000.0,  # 10 Lakh INR
                 max_position_size: float = 0.2,   # Max 20% in single stock
                 exchange: str = "NSE",  # NSE or BSE
                 data_root: str = "market_data",
                 stocks: Optional[List[str]] = None):
        
        self.initial_cash = initial_cash
        self.max_position_size = max_position_size
        self.exchange = exchange
        self.data_root = Path(data_root)
        
        # Portfolio state
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.transactions: List[Transaction] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.start_date = None
        self.current_date = None
        self.daily_returns = []
        self.max_drawdown = 0.0
        self.peak_value = initial_cash
        
        # --- TAX / CAPITAL GAINS bookkeeping ---
        # Total realized short-term and long-term gains
        self.total_stcg = 0.0
        self.total_ltcg = 0.0
        # Total tax paid so far
        self.total_tax_paid = 0.0
        # STCG rate change effective date (23 Jul 2024) as you referenced
        self.stcg_rate_change_date = datetime(2024, 7, 23)
        self.stcg_rate_before = 0.15   # 15% for transfers before change date
        self.stcg_rate_after = 0.20    # 20% for transfers on/after change date
        self.cess_rate = 0.04          # 4% health & education cess
        # Optional surcharge (set to 0.0 unless you want to simulate)
        self.surcharge_rate = 0.0

        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load stock list
        self.stocks = stocks if stocks else self._load_stock_list()
        
        # Market data cache
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
        
    def _load_stock_list(self) -> List[str]:
        """Load list of stocks from stocks.txt"""
        stocks_file = Path("finagent/stocks.txt")
        if stocks_file.exists():
            with open(stocks_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        return []
    
    def _generate_transaction_id(self) -> str:
        """Generate unique transaction ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"TXN_{timestamp}"
    
    def calculate_indian_trading_costs(self, transaction_value: float, is_buy: bool = True) -> Dict[str, float]:
        """
        Calculate Indian stock trading costs based on equity delivery.
        
        Args:
            transaction_value: Value of the transaction (quantity * price)
            is_buy: True for buy transactions, False for sell
            
        Returns:
            Dictionary with breakdown of all costs
        """
        costs = {}
        
        # Brokerage: Zero
        costs['brokerage'] = 0.0
        
        # STT/CTT: 0.1% on buy & sell
        costs['stt'] = transaction_value * 0.001
        
        # Transaction charges (depends on exchange)
        if self.exchange.upper() == "NSE":
            costs['transaction_charges'] = transaction_value * 0.00297 / 100
        else:  # BSE
            costs['transaction_charges'] = transaction_value * 0.00375 / 100
        
        # SEBI charges: ₹10 / crore
        costs['sebi_charges'] = (transaction_value / 10000000) * 10  # 1 crore = 10^7
        
        # Stamp charges: 0.015% or ₹1500 / crore on buy side only
        if is_buy:
            stamp_percentage = transaction_value * 0.00015
            stamp_flat = (transaction_value / 10000000) * 1500
            costs['stamp_charges'] = min(stamp_percentage, stamp_flat)
        else:
            costs['stamp_charges'] = 0.0
        
        # GST: 18% on (brokerage + SEBI charges + transaction charges)
        taxable_amount = costs['brokerage'] + costs['sebi_charges'] + costs['transaction_charges']
        costs['gst'] = taxable_amount * 0.18
        
        # Total costs
        costs['total_costs'] = sum(costs.values())
        
        return costs
    
    def get_max_buyable_shares(self, price: float) -> int:
        """
        Calculate maximum whole shares that can be bought with available cash.
        
        Args:
            price: Price per share
            
        Returns:
            Maximum whole shares that can be bought
        """
        if price <= 0:
            return 0
        
        # Binary search to find maximum shares considering transaction costs
        max_shares = int(self.cash / price)  # Upper bound
        min_shares = 0
        
        while min_shares <= max_shares:
            shares = (min_shares + max_shares) // 2
            transaction_value = shares * price
            costs = self.calculate_indian_trading_costs(transaction_value, is_buy=True)
            total_required = transaction_value + costs['total_costs']
            
            if total_required <= self.cash:
                min_shares = shares + 1
            else:
                max_shares = shares - 1
        
        return max_shares

    def _add_buy_lot(self, stock_symbol: str, quantity: int, price: float, fees: float, timestamp: datetime):
        """Append a buy lot to the position (create position if not exists)"""
        lot = {
            'quantity': int(quantity),
            'price': float(price),
            'entry_date': timestamp,
            'fees': float(fees)
        }
        if stock_symbol in self.positions:
            position = self.positions[stock_symbol]
            position.lots.append(lot)
            position.current_price = price
            position.last_updated = timestamp
        else:
            position = Position(
                stock_symbol=stock_symbol,
                side=PositionSide.LONG,
                lots=[lot],
                current_price=price,
                last_updated=timestamp
            )
            self.positions[stock_symbol] = position

    def _match_sell_lots(self, stock_symbol: str, sell_quantity: int, sell_price: float,
                         sell_costs: Dict[str, float], timestamp: datetime) -> Dict[str, Any]:
        """
        Match sell_quantity to existing lots FIFO, compute realized gains per lot,
        classify into STCG / LTCG, and return summary dict:
          {
            'stcg': <amount>,
            'ltcg': <amount>,
            'tax': <computed_tax>,
            'gain_details': [ {lot_qty, lot_price, buy_fees, sell_proceeds, holding_days, gain} ... ]
          }
        """
        result = {'stcg': 0.0, 'ltcg': 0.0, 'tax': 0.0, 'gain_details': []}
        if stock_symbol not in self.positions:
            return result

        position = self.positions[stock_symbol]
        remaining = int(sell_quantity)
        total_tx_value = sell_quantity * sell_price
        sell_fees_total = float(sell_costs.get('total_costs', 0.0))

        # Iterate lots FIFO
        while remaining > 0 and position.lots:
            lot = position.lots[0]
            lot_qty = int(lot['quantity'])
            take_qty = min(lot_qty, remaining)

            buy_price = float(lot['price'])
            buy_fees = float(lot.get('fees', 0.0))
            # allocate buy fees proportionally to quantity taken
            buy_fee_alloc = (take_qty / lot_qty) * buy_fees if lot_qty > 0 else 0.0

            sell_proceeds = take_qty * sell_price
            # allocate sell fees proportionally across the sell transaction value
            sell_fee_alloc = (sell_proceeds / total_tx_value) * sell_fees_total if total_tx_value > 0 else 0.0

            cost_basis = (take_qty * buy_price) + buy_fee_alloc
            gain = sell_proceeds - cost_basis - sell_fee_alloc

            holding_days = (timestamp.date() - lot['entry_date'].date()).days if lot.get('entry_date') else None

            # classify
            if holding_days is not None and holding_days <= 365:
                result['stcg'] += gain
            else:
                result['ltcg'] += gain

            result['gain_details'].append({
                'lot_qty': take_qty,
                'lot_price': buy_price,
                'buy_fee_alloc': buy_fee_alloc,
                'sell_fee_alloc': sell_fee_alloc,
                'sell_proceeds': sell_proceeds,
                'holding_days': holding_days,
                'gain': gain
            })

            # reduce or pop lot
            if take_qty == lot_qty:
                position.lots.pop(0)
            else:
                lot['quantity'] = lot_qty - take_qty

            remaining -= take_qty

        # If remaining > 0, user tried to sell more than lots (should be prevented by can_sell)
        # Compute tax for STCG only (LTCG handling left in ledger)
        stcg_amount = result['stcg']
        tax_on_stcg = 0.0
        if stcg_amount > 0:
            rate = self.stcg_rate_after if timestamp.date() >= self.stcg_rate_change_date.date() else self.stcg_rate_before
            tax_on_stcg = stcg_amount * rate
            # add surcharge if any
            if self.surcharge_rate:
                tax_on_stcg += tax_on_stcg * self.surcharge_rate
            # cess
            tax_on_stcg += tax_on_stcg * self.cess_rate

        result['tax'] = tax_on_stcg
        return result

    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get current portfolio state for RL environment.
        
        Returns:
            Dictionary containing portfolio state information
        """
        total_value = self.get_total_portfolio_value()
        
        # Calculate position weights
        position_weights = {}
        for symbol, position in self.positions.items():
            if total_value > 0:
                position_weights[symbol] = position.market_value / total_value
            else:
                position_weights[symbol] = 0.0
        
        # Calculate sector exposure (if sector data available)
        sector_exposure = self._calculate_sector_exposure()
        
        state = {
            'cash': self.cash,
            'total_value': total_value,
            'cash_weight': self.cash / total_value if total_value > 0 else 1.0,
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'side': pos.side.value,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_percent': pos.unrealized_pnl_percent,
                    'weight': position_weights.get(symbol, 0.0)
                }
                for symbol, pos in self.positions.items()
            },
            'position_weights': position_weights,
            'sector_exposure': sector_exposure,
            'num_positions': len(self.positions),
            'max_drawdown': self.max_drawdown,
            'total_return': self.get_total_return(),
            'sharpe_ratio': self.get_sharpe_ratio(),
            'volatility': self.get_volatility()
        }
        
        return state
    
    def get_total_portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + positions)"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_total_return(self) -> float:
        """Calculate total return since inception"""
        if self.initial_cash == 0:
            return 0.0
        return ((self.get_total_portfolio_value() - self.initial_cash) / self.initial_cash) * 100
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio"""
        if len(self.daily_returns) < 2:
            return 0.0
        
        returns_array = np.array(self.daily_returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def get_volatility(self) -> float:
        """Calculate portfolio volatility"""
        if len(self.daily_returns) < 2:
            return 0.0
        
        returns_array = np.array(self.daily_returns)
        return np.std(returns_array) * np.sqrt(252) * 100  # Annualized volatility
    
    def _calculate_sector_exposure(self) -> Dict[str, float]:
        """Calculate sector exposure based on positions"""
        # This would need sector mapping data
        # For now, return empty dict
        return {}
    
    def can_buy(self, stock_symbol: str, quantity: int, price: float) -> Tuple[bool, str]:
        """
        Check if a buy order can be executed.
        
        Args:
            stock_symbol: Stock to buy
            quantity: Quantity to buy (whole shares)
            price: Price per share
            
        Returns:
            Tuple of (can_execute, reason)
        """
        # Ensure whole shares
        if not isinstance(quantity, int) or quantity <= 0:
            return False, f"Quantity must be a positive whole number. Received: {quantity}"
        
        transaction_value = quantity * price
        costs = self.calculate_indian_trading_costs(transaction_value, is_buy=True)
        required_cash = transaction_value + costs['total_costs']
        
        if required_cash > self.cash:
            return False, f"Insufficient cash. Required: ₹{required_cash:.2f}, Available: ₹{self.cash:.2f}"
        
        # Check position size limits
        if stock_symbol in self.positions:
            current_position = self.positions[stock_symbol]
            new_position_value = (current_position.quantity + quantity) * price
            total_portfolio_value = self.get_total_portfolio_value()
            
            if new_position_value / total_portfolio_value > self.max_position_size:
                return False, f"Position size limit exceeded. Max: {self.max_position_size*100}%"
        
        return True, "Order can be executed"
    
    def can_sell(self, stock_symbol: str, quantity: int) -> Tuple[bool, str]:
        """
        Check if a sell order can be executed.
        
        Args:
            stock_symbol: Stock to sell
            quantity: Quantity to sell (whole shares)
            
        Returns:
            Tuple of (can_execute, reason)
        """
        # Ensure whole shares
        if not isinstance(quantity, int) or quantity <= 0:
            return False, f"Quantity must be a positive whole number. Received: {quantity}"
        
        if stock_symbol not in self.positions:
            return False, f"No position in {stock_symbol}"
        
        position = self.positions[stock_symbol]
        if position.side != PositionSide.LONG:
            return False, f"Cannot sell short position in {stock_symbol}"
        
        if quantity > position.quantity:
            return False, f"Insufficient shares. Requested: {quantity}, Available: {position.quantity}"
        
        return True, "Order can be executed"
    
    def execute_buy(self, stock_symbol: str, quantity: int, price: float, 
                    timestamp: datetime = None, notes: str = "") -> bool:
        """
        Execute a buy order.
        
        Args:
            stock_symbol: Stock to buy
            quantity: Quantity to buy (whole shares)
            price: Price per share
            timestamp: Transaction timestamp
            notes: Additional notes
            
        Returns:
            True if successful, False otherwise
        """
        can_buy, reason = self.can_buy(stock_symbol, quantity, price)
        if not can_buy:
            self.logger.warning(f"Buy order rejected: {reason}")
            return False
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate Indian trading costs
        transaction_value = quantity * price
        costs = self.calculate_indian_trading_costs(transaction_value, is_buy=True)
        total_cost = transaction_value + costs['total_costs']
        
        # Deduct cash
        self.cash -= total_cost
        
        # Record buy lot for FIFO bookkeeping
        self._add_buy_lot(stock_symbol=stock_symbol,
                          quantity=quantity,
                          price=price,
                          fees=costs['total_costs'],
                          timestamp=timestamp)
        
        # Record transaction
        transaction = Transaction(
            transaction_id=self._generate_transaction_id(),
            stock_symbol=stock_symbol,
            transaction_type="BUY",
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            fees=costs['total_costs'],
            notes=f"{notes} | Costs: STT=₹{costs['stt']:.2f}, TC=₹{costs['transaction_charges']:.2f}, Stamp=₹{costs['stamp_charges']:.2f}, GST=₹{costs['gst']:.2f}"
        )
        self.transactions.append(transaction)
        
        self.logger.info(f"Buy order executed: {quantity} {stock_symbol} @ ₹{price:.2f} (Total cost: ₹{total_cost:.2f})")
        return True
    
    def execute_sell(self, stock_symbol: str, quantity: int, price: float,
                     timestamp: datetime = None, notes: str = "") -> bool:
        """
        Execute a sell order with FIFO matching to compute realized gains and STCG tax.
        """
        can_sell, reason = self.can_sell(stock_symbol, quantity)
        if not can_sell:
            self.logger.warning(f"Sell order rejected: {reason}")
            return False

        if timestamp is None:
            timestamp = datetime.now()

        # Calculate Indian trading costs (sell side)
        transaction_value = quantity * price
        costs = self.calculate_indian_trading_costs(transaction_value, is_buy=False)
        total_proceeds = transaction_value - costs['total_costs']

        # Match lots FIFO and compute realized gains & tax
        match_result = self._match_sell_lots(stock_symbol=stock_symbol,
                                             sell_quantity=quantity,
                                             sell_price=price,
                                             sell_costs=costs,
                                             timestamp=timestamp)

        stcg = match_result.get('stcg', 0.0)
        ltcg = match_result.get('ltcg', 0.0)
        tax = match_result.get('tax', 0.0)

        # Update overall ledger totals
        self.total_stcg += stcg
        self.total_ltcg += ltcg
        self.total_tax_paid += tax

        # Now update cash: add proceeds then subtract tax
        self.cash += total_proceeds
        if tax > 0:
            # Deduct tax immediately
            self.cash -= tax
            # Record tax as separate transaction (for audit)
            tax_tx = Transaction(
                transaction_id=self._generate_transaction_id(),
                stock_symbol=stock_symbol,
                transaction_type="TAX",
                quantity=0,
                price=0.0,
                timestamp=timestamp,
                fees=tax,
                notes=f"STCG tax deducted on realized STCG ₹{stcg:.2f} = tax ₹{tax:.2f}"
            )
            self.transactions.append(tax_tx)

        # Update position (quantity and current price)
        position = self.positions.get(stock_symbol)
        if position:
            position.current_price = price
            position.last_updated = timestamp
            # If all lots consumed, remove position
            if position.quantity == 0:
                del self.positions[stock_symbol]

        # Record sell transaction
        transaction = Transaction(
            transaction_id=self._generate_transaction_id(),
            stock_symbol=stock_symbol,
            transaction_type="SELL",
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            fees=costs['total_costs'],
            notes=f"{notes} | Costs: STT=₹{costs['stt']:.2f}, TC=₹{costs['transaction_charges']:.2f}, GST=₹{costs['gst']:.2f} | Realized STCG=₹{stcg:.2f}, LTCG=₹{ltcg:.2f}, TAX=₹{tax:.2f}"
        )
        self.transactions.append(transaction)

        self.logger.info(f"Sell order executed: {quantity} {stock_symbol} @ ₹{price:.2f} (Net proceeds: ₹{total_proceeds:.2f}) Realized STCG: ₹{stcg:.2f}, TAX deducted: ₹{tax:.2f}")
        return True

    def get_tax_summary(self) -> Dict[str, float]:
        """Return a quick summary of realized gains and taxes paid"""
        return {
            'total_stcg': self.total_stcg,
            'total_ltcg': self.total_ltcg,
            'total_tax_paid': self.total_tax_paid
        }
    
    def update_market_prices(self, market_data: Dict[str, float], timestamp: datetime = None):
        """
        Update current market prices for all positions.
        
        Args:
            market_data: Dictionary of stock_symbol -> current_price
            timestamp: Update timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Update current date if provided
        if self.current_date is None:
            self.current_date = timestamp.date()
        elif timestamp.date() != self.current_date:
            # New trading day
            self._record_daily_snapshot()
            self.current_date = timestamp.date()
        
        # Update position prices
        for stock_symbol, price in market_data.items():
            if stock_symbol in self.positions:
                self.positions[stock_symbol].current_price = price
                self.positions[stock_symbol].last_updated = timestamp
        
        # Update portfolio history
        self._update_portfolio_history(timestamp)
    
    def _record_daily_snapshot(self):
        """Record daily portfolio snapshot"""
        if self.current_date is None:
            return
        
        snapshot = {
            'date': self.current_date,
            'cash': self.cash,
            'total_value': self.get_total_portfolio_value(),
            'num_positions': len(self.positions),
            'total_return': self.get_total_return(),
            'max_drawdown': self.max_drawdown
        }
        
        self.portfolio_history.append(snapshot)
        
        # Calculate daily return
        if len(self.portfolio_history) > 1:
            prev_value = self.portfolio_history[-2]['total_value']
            current_value = snapshot['total_value']
            daily_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0
            self.daily_returns.append(daily_return)
            
            # Update max drawdown
            if current_value > self.peak_value:
                self.peak_value = current_value
            
            drawdown = (self.peak_value - current_value) / self.peak_value
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
    
    def _update_portfolio_history(self, timestamp: datetime):
        """Update real-time portfolio history"""
        current_value = self.get_total_portfolio_value()
        
        # Update peak value and drawdown
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        drawdown = (self.peak_value - current_value) / self.peak_value
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def get_position_summary(self) -> pd.DataFrame:
        """Get summary of all positions as DataFrame"""
        if not self.positions:
            return pd.DataFrame()
        
        summary_data = []
        for symbol, position in self.positions.items():
            summary_data.append({
                'Stock': symbol,
                'Side': position.side.value,
                'Quantity': position.quantity,
                'Entry_Price': position.entry_price,
                'Current_Price': position.current_price,
                'Market_Value': position.market_value,
                'Unrealized_PnL': position.unrealized_pnl,
                'Unrealized_PnL_%': position.unrealized_pnl_percent,
                'Entry_Date': position.entry_date,
                'Last_Updated': position.last_updated
            })
        
        return pd.DataFrame(summary_data)
    
    def get_transaction_history(self) -> pd.DataFrame:
        """Get transaction history as DataFrame"""
        if not self.transactions:
            return pd.DataFrame()
        
        transaction_data = []
        for tx in self.transactions:
            transaction_data.append({
                'Transaction_ID': tx.transaction_id,
                'Stock': tx.stock_symbol,
                'Type': tx.transaction_type,
                'Quantity': tx.quantity,
                'Price': tx.price,
                'Total_Value': tx.total_value,
                'Fees': tx.fees,
                'Timestamp': tx.timestamp,
                'Notes': tx.notes
            })
        
        return pd.DataFrame(transaction_data)
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """Get portfolio history as DataFrame"""
        if not self.portfolio_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.portfolio_history)
    
    def save_portfolio_state(self, file_path: str = "portfolio_state.json"):
        """Save current portfolio state to JSON file"""
        state = {
            'cash': self.cash,
            'initial_cash': self.initial_cash,
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'side': pos.side.value,
                    'entry_price': pos.entry_price,
                    'entry_date': pos.entry_date.isoformat(),
                    'current_price': pos.current_price,
                    'last_updated': pos.last_updated.isoformat()
                }
                for symbol, pos in self.positions.items()
            },
            'transactions': [
                {
                    'transaction_id': tx.transaction_id,
                    'stock_symbol': tx.stock_symbol,
                    'transaction_type': tx.transaction_type,
                    'quantity': tx.quantity,
                    'price': tx.price,
                    'timestamp': tx.timestamp.isoformat(),
                    'fees': tx.fees,
                    'notes': tx.notes
                }
                for tx in self.transactions
            ],
            'portfolio_history': self.portfolio_history,
            'daily_returns': self.daily_returns,
            'max_drawdown': self.max_drawdown,
            'peak_value': self.peak_value
        }
        
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.logger.info(f"Portfolio state saved to {file_path}")
    
    def load_portfolio_state(self, file_path: str = "portfolio_state.json"):
        """Load portfolio state from JSON file"""
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # Restore basic state
            self.cash = state.get('cash', self.initial_cash)
            self.max_drawdown = state.get('max_drawdown', 0.0)
            self.peak_value = state.get('peak_value', self.initial_cash)
            self.daily_returns = state.get('daily_returns', [])
            self.portfolio_history = state.get('portfolio_history', [])
            
            # Restore positions
            self.positions.clear()
            for symbol, pos_data in state.get('positions', {}).items():
                # Reconstruct lots from saved data (simplified for basic restoration)
                lots = [{
                    'quantity': pos_data['quantity'],
                    'price': pos_data['entry_price'],
                    'entry_date': datetime.fromisoformat(pos_data['entry_date']),
                    'fees': 0.0  # Default since not saved in old format
                }]
                
                position = Position(
                    stock_symbol=symbol,
                    side=PositionSide(pos_data['side']),
                    lots=lots,
                    current_price=pos_data['current_price'],
                    last_updated=datetime.fromisoformat(pos_data['last_updated'])
                )
                self.positions[symbol] = position
            
            # Restore transactions
            self.transactions.clear()
            for tx_data in state.get('transactions', []):
                transaction = Transaction(
                    transaction_id=tx_data['transaction_id'],
                    stock_symbol=tx_data['stock_symbol'],
                    transaction_type=tx_data['transaction_type'],
                    quantity=tx_data['quantity'],
                    price=tx_data['price'],
                    timestamp=datetime.fromisoformat(tx_data['timestamp']),
                    fees=tx_data['fees'],
                    notes=tx_data['notes']
                )
                self.transactions.append(transaction)
            
            self.logger.info(f"Portfolio state loaded from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading portfolio state: {e}")
    
    def reset_portfolio(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_cash
        self.positions.clear()
        self.transactions.clear()
        self.portfolio_history.clear()
        self.daily_returns.clear()
        self.max_drawdown = 0.0
        self.peak_value = self.initial_cash
        self.start_date = None
        self.current_date = None
        
        self.logger.info("Portfolio reset to initial state")