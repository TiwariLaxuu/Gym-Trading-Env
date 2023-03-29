# Crypto-Trading-Env

An OpenAI Gym environment for simulating stocks and train Reinforcement Learning trading agents.

Designed to be **FAST** and **CUSTOMIZABLE** for an easy RL trading algorythms implementation.
## Install and import
```pip install gym-trading-env```

Then import :

```python
from gym_trading_env import TradingEnv
```

## Environment Properties

### Actions space : positions

Github is full of environments that consider actions such as **BUY**, **SELL**. In my opinion, it is a real mistake to think a RL-agent as a trader. Traders make trade and to do so, they place orders on the market (eg. Buy X of stock Y). But what really matter is the position reached. For example, a trader that sell half of his stocks Y, wants to reduce his risk, but also his potential gains. Now, imagine we labelled each position by a number :
- ```1``` : We have bought as much as possible of stock Y (LONG); ideally, we have all of our stock's portfolio converted in the stock Y
- ```0``` : We have sold as much as possible of stock Y (OUT); ideally, we have all of our stock's portfolio converted in our currency
Now, we can imagine half position or others :
- ```0.5``` : 50% in stock Y & 50% in currency
- Even : ```0.1``` : 10% in stock Y & 90% in currency
....


In fact, it is way simpler for a RL-agent to work with positions. This way, it can easily make complex operation with a simple action space.
Plus, this environment supports more complex positions such as:
- ```-1``` : once every stock Y is sold, we bet 100% of the portfolio value on the decline of asset Y. To perform this action, the environment borrows 100% of the portfolio valuation as stock Y to an imaginary person, and immediately sell it. When the agent closes the position (position 0), the environment buys the owed amount of stock Y and repays the imaginary person with it. If the price has fallen during the operation, we buy cheaper than we sold what we need to repay : the difference is our gain. The imaginary person is paid a small rent (parameter : ```borrow_interest_rate```)
- ```+2``` : buy as much stock as possible, then we bet 100% of the portfolio value of the rise of asset Y. We use the same mechanism explained above, but we borrow currency and buy stock Y.
- ```-10``` ? : We can BUT ...  We need to borrow 1000% of the portfolio valuation as asset Y. You need to understand that such a "leverage" is very risky. As if the stock price rise by 10%, you need to repay the original 1000% of your portfolio valuation at 1100% (1000%*1.10) portfolio valuation. Well, 100% (1100% - 1000%) of your portfolio is used to repay your debt. **GAME OVER, you have 0$ left**. The leverage is very useful but also risky, as it increases your **gains** AND your **losses**. Always keep in mind that you can lose everything.

### How to use ?

**1 - Import and clean your data**. They need to be ordered by ascending date. Index must be a date.
```python
df = pd.read_csv("data/BTC_USD-Hourly.csv", parse_dates=["date"], index_col= "date")
df.sort_index(inplace= True)
df.dropna(inplace= True)
df.drop_duplicates(inplace=True)s
```
**2 - Create your feature**. Your RL-agent will need some good, preprocessed features. It is your job to make sure it has everything it needs.
**The feature column names need to contain the keyword 'feature'. The environment will automatically detect them !**

```python
df["feature_close"] = df["close"].pct_change()
df["feature_open"] = df["open"]/df["close"]
df["feature_high"] = df["high"]/df["close"]
df["feature_low"] = df["low"]/df["close"]
df["feature_volume"] = df["Volume USD"] / df["Volume USD"].rolling(7*24).max()
df.dropna(inplace= True) # Clean your data !
```
**(Optional)3 - Create your own reward function**. Use the history object described below to create your own !
```python
import numpy as np
def reward_function(history):
    return np.log(history[-1]["portfolio_valuation"] / history[-2]["portfolio_valuation"]) #log (p_t / p_t-1 )

>>> output : history[t] # data history at step t
{
    "step": ...,# Step = t
    "date": ...,# Date at step t
    "reward": ..., # Reward at step t
    "position_index": ..., # Index of the position at step t amoung your position argument
    "position" : ..., # Portfolio position at step t
    "df_info": { # Gather every data at step t from your DataFrame's columns, that are not features
        "none_feature_columns1":...,
        "none_feature_columns2":...,
        "none_feature_columns3":..., 
        .... # For example : open, high, low, close,
    },
    "portfolio_valuation": ..., # Valuation of the portfolio at step t
    "portfolio_distribution":{
            "asset" : ...,
            "fiat" : ...,
            "borrowed_asset": ...,
            "borrowed_fiat": ...,
            "interest_asset": ...,
            "interest_fiat": ...,
    }
}
```

**4 - Create the environment**
```python
env = TradingEnv(
    df = df,
    windows= 5, # Windows, default : None. If None, observation at t are the features at step t. If windows = i (int),  observation at t are the features from steps [t-i+1 :  t]
    positions = [-1, -0.5, 0, 0.5, 1], # Positions, default : [0, 1], that the agent can choose (Explained in "Actions space : positions")
    initial_position = 0, #Initial position, default = 0
    trading_fees = 0.01/100, # Trading fee, default : 0. Here, 0.01% per stock buy / sell)
    borrow_interest_rate= 0.0003/100, # Borrow interest rate PER STEP, default : 0. Here we pay 0.0003% per HOUR per asset borrowed
    reward_function = reward_function, # Reward function, default : the one presented above
    portfolio_initial_value = 1000, # Initial value of the portfolio (in FIAT), default : 1000. Here, 1000 USD
)
```
**5 - Run the environment**
```python
truncated = False
observation, info = env.reset()
while not truncated:
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
```
- ```observation``` returns a dict with items :
    - ```features``` : Contains the features created. If windows is None, it contains the features of the current step (shape = (n_features,)). If windows is i (int), it contains the features the last i steps (shape = (5, n_features)).
    - ```position``` : The last position of the environments. It can be useful to include this to the features, so the agent knows which position he is holding and gains stability and continuity.
- ```reward``` : The step reward following the action taken.
- ```done```: Always False.
- ```truncated``` : Is true if we reached the end of the DataFrame.
- ```info``` : Return the last history step of the object "history" presented above (in "3 - Create your own reward function")


**(Optional) 6 - Render**

Performed with Dash Plotly (local app).
```python
env.render()
```
<img alt="Render example" src ="https://github.com/ClementPerroud/Gym-Trading-Env/blob/main/readme_images/render.PNG?raw=true" height = "600"/>

Enjoy :)




