class Strategy:
    def __init__(self, data):
        self.data = data
        self.positions = []

    def generate_signals(self):
        raise NotImplementedError("Subclasses must implement this method.")

class RegimeBasedStrategy(Strategy):
    def generate_signals(self):
        signals = []
        for i in range(len(self.data)):
            state = self.data.iloc[i]['hidden_states']
            if state == 0:
                signals.append(1)   # long
            elif state == 1:
                signals.append(-1)  # short
            else:
                signals.append(0)   # flat
        self.data['signal'] = signals
        return self.data

class CustomRegimeBasedStrategy(RegimeBasedStrategy):
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        signals = []
        for i in range(len(self.data)):
            # Use the 'returns' or 'close' column instead of 'hidden_states'
            returns = self.data.iloc[i]['returns']  # Assuming 'returns' is available for signal generation
            volatility = self.data.iloc[i]['volatility']  # Assuming 'volatility' is available for volatility checks

            # Example signal generation based on returns and volatility:
            if returns > 0 and volatility < self.data['volatility'].median():
                signals.append(1)  # Buy (long)
            elif returns < 0 and volatility > self.data['volatility'].median():
                signals.append(-1)  # Sell (short)
            else:
                signals.append(0)  # Hold (flat)

        self.data['signal'] = signals
        return self.data