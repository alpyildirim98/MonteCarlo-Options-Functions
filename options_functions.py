#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:35:01 2023

@author: alpyildirim
"""



import datetime as dtime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import math

def MonteCarloEngine(S, r, sigma, T, steps, simulations, seed = 42):
    """
    Simulates stock price paths using Monte Carlo simulation.
    
    Parameters
    ----------
    S : int
        Initial stock price
    r : float
        Annualized risk-free rate
    sigma : float
        Annualized volatility
    T : int
        The time to expiry(in years)
    steps : int
        Number of time steps in a time period
    simulations : int
        Number of simulated paths
    seed : int, optional
        Seed for the random number generator. Default is 42.

    Returns
    -------
    spot_prices : A 2 dimensional numpy array of shape (steps + 1, paths)
    Contains stock prices simulated using Geometric Brownian Motion    
    """       
    
    np.random.seed(seed)    
    dt = T/steps
    spot_prices = np.zeros((simulations, steps))
    spot_prices[:, 0] = S
        
    for i in range(1, steps):
        eps = np.random.normal(0, 1, simulations)
        spot_prices[:, i] = spot_prices[:, i-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * eps)
            
    return spot_prices

def BlackScholes_d1(S, K, r, T, sigma):
    """
    Returns the Black-Scholes d1 value given the underlying stock price, strike price,
    annualized risk-free rate, time to expiry and annualized volatility.
    
    Parameters
    ----------
    S : float
        Price of the underlying asset
    K : int
        Strike price
    r : float
        Annualized risk-free rate
    T : float
        The time to expiry
    sigma: float
        Annualized volatility of the underlying asset

    Returns
    -------
    d1 : float
        The Black-Scholes d1 value
    """  
    
    d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T) * 1/(sigma * T ** 0.5)
        
    return d1

    
def BlackScholes_d2(d1, T, sigma):
    """
    Returns the Black-Scholes d2 value according to the calculated d1 level.
    
    Parameters
    ----------
    d1 : float
        The Black-Scholes d1 value returned from the BlackScholes_d1 function
    T : float
        The time to expiry
    sigma:
        Annualized volatility of the underlying asset

    Returns
    -------
    d2 : float
        The Black-Scholes d2 value
    """
    
    d2 = d1 - sigma * T ** 0.5
    
    return d2

def calculate_delta(d1):
    """
    Calculates the delta value for an option given the Black-Scholes value d1.
    
    Parameters
    ----------
    d1 : float
        The Black-Scholes d1 value returned from the BlackScholes_d1 function
        
    Returns
    -------
    delta : float
        The delta value for the option
    """
    
    delta = stats.norm.cdf(d1)
    return delta

def calculate_gamma(d1, S, sigma, T):
    """
    Calculates the gamma value for an option given the Black-Scholes value d1, spot price, volatility and
    time to expiry.

    Parameters
    ----------
    d1 : float
        The Black-Scholes d1 value returned from the BlackScholes_d1 function
    S : float
        Price of the underlying asset
    sigma : float
        Annualized volatility of the underlying asset
    T : float
        The time to expiry

    Returns
    -------
    gamma : float
        The gamma value for the option.

    """
    
    gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

def calculate_call_price(spot_prices, K, r, iv, T, steps, rv = None):  
    """
    Calculates the prices, deltas, and gammas of European call options using the Black-Scholes model.
    Uses implied volatility for calculations if realized volatility is not specified.

    Parameters
    ----------
    spot_prices : float or ndarray
        Current prices of the underlying asset
    K : int
        Strike price of the option
    r : float
        Annualized risk-free rate
    iv : float
        Implied volatility of the option
    T : float
        Time to expiration of the option
    steps : int
        Number of time steps to use in the calculation
    rv : float, optional
        Realized volatility of the underlying asset. The default is None

    Returns
    -------
    call_prices : A 2 dimensional ndarray, float
        The call prices of the option at each time step
    deltas : A 2 dimensional ndarray, float
        The delta values of the option at each time step
    gammas : A 2 dimensional ndarray, float
        The gamma values of the option at each time step

    """
    
    times = np.linspace(T/steps, T, (steps * T))[::-1]
    
    if rv is None:
        d1 = BlackScholes_d1(spot_prices, K, r, times, iv)
        d2 = BlackScholes_d2(d1, times, iv)
        deltas = calculate_delta(d1)
        gammas = calculate_gamma(d1, spot_prices, iv, times)
        np.nan_to_num(gammas, 0)
        call_prices = deltas * spot_prices - stats.norm.cdf(d2) * K * np.exp(-r * times)
    
    else:
        d1 = BlackScholes_d1(spot_prices, K, r, times, rv)
        d2 = BlackScholes_d2(d1, times, rv)
        deltas = calculate_delta(d1)
        gammas = calculate_gamma(d1, spot_prices, rv, times)
        np.nan_to_num(gammas, 0)
        call_prices = deltas * spot_prices - stats.norm.cdf(d2) * K * np.exp(-r * times)
        
    return call_prices, deltas, gammas


def compute_pnl_no_tcost(spot_prices, gammas, rv, iv, steps, T):
    """
    Computes the running PnL of the delta-hedging strategy in the absence of transaction costs.

    Parameters
    ----------
    spot_prices : ndarray
       Current prices of the underlying asset 
    gammas : ndarray
        The gamma values of the option at each time step
    rv : float
        The realized volatility of the underlying asset
    iv : float
        The implied volatility of the option
    steps : int
        Number of time steps to use in the calculation
    T : float
        Time to expiration of the option

    Returns
    -------
    running_pnl_no_tcost : ndarray
        The running PnL of the delta-hedging strategy

    """
    
    dt = np.full((steps * T), 1/(steps*T))
    dt[0] = 0
    pnl = 0.5 * (rv - iv) * spot_prices**2 * gammas * dt
    running_pnl = np.cumsum(pnl, axis = 1)
    running_pnl_no_tcost = np.delete(running_pnl, -1, axis=1)
    
    return running_pnl_no_tcost

def compute_pnl_tcost(spot_prices, deltas, gammas, rv, iv, steps, T, transaction_cost, delta_changes = None):
    """
    Computes the running PnL of the delta-hedging strategy when the transaction costs are present.

    Parameters
    ----------
    spot_prices : ndarray
       Current prices of the underlying asset 
    deltas : ndarray
        The delta values of the option at each time step
    gammas : ndarray
        The gamma values of the option at each time step
    rv : float
        The realized volatility of the underlying asset
    iv : float
        The implied volatility of the option
    steps : int
        Number of time steps to use in the calculation
    T : float
        Time to expiration of the option
    transaction_cost : float
        The cost of buying and selling the stock to rehedge the delta.
    delta_changes = ndarray
        The changes in the delta value used to calculate the transaction costs.
        If not specified, the delta changes are calculated in each step automatically.
    
    Returns
    -------
    running_pnl_tcost : ndarray
        The running PnL of the delta-hedging strategy in the presence of transaction costs.

    """
    if delta_changes is None:
        dt = np.full((steps * T), 1/(steps*T))
        dt[0] = 0
        d_delta = np.diff(deltas, axis=1)
        d_delta = np.insert(d_delta, 0, deltas[:,0], axis=1)
        pnl = (0.5 * (rv - iv) * spot_prices**2 * gammas * dt) - (abs(d_delta) * spot_prices * transaction_cost)
        running_pnl = np.cumsum(pnl, axis = 1)
        running_pnl_tcost = np.delete(running_pnl, -1, axis=1)
        
    else:
        dt = np.full((steps * T), 1/(steps*T))
        dt[0] = 0
        delta_changes = delta_changes[:,1:]
        delta_changes = np.insert(delta_changes, 0, deltas[:,0], axis=1)
        pnl = (0.5 * (rv - iv) * spot_prices**2 * gammas * dt) - (delta_changes * spot_prices * transaction_cost)
        running_pnl = np.cumsum(pnl, axis = 1)
        running_pnl_tcost = np.delete(running_pnl, -1, axis=1)
    
    return running_pnl_tcost

def compute_pnl_time_based(spot_prices, deltas, gammas, rv, iv, steps, T, transaction_cost, rehedge_freq):
    """
    Computes the running PnL of time-based delta-hedging strategies according to the given rehedge frequency.

    Parameters
    ----------
    spot_prices : ndarray
       Current prices of the underlying asset 
    deltas : ndarray
        The delta values of the option at each time step
    gammas : ndarray
        The gamma values of the option at each time step
    rv : float
        The realized volatility of the underlying asset
    iv : float
        The implied volatility of the option
    steps : int
        Number of time steps to use in the calculation
    T : float
        Time to expiration of the option
    transaction_cost : float
        The cost of buying and selling the stock to rehedge the delta.
    rehedge_freq : int
        The number of steps between each rehedging operation.

    Returns
    -------
    running_pnl_tcost : ndarray
        The running PnL of the delta-hedging strategy in the presence of transaction costs.

    """
   
    dt = np.full((steps * T,), 1/(steps*T))
    dt[0] = 0
    
    delta_chgs = np.zeros((10000, 1000))
    delta_chgs = np.delete(delta_chgs, 0, axis = 1)
    delta_chgs = np.insert(delta_chgs, 0, deltas[:,0], axis=1)
        
    for i in range(rehedge_freq, steps, rehedge_freq):
        delta_chgs[:, i] = abs(deltas[:, i] - deltas[:, i-rehedge_freq])
    
    pnl = (0.5 * (rv - iv) * spot_prices**2 * gammas * dt) - (abs(delta_chgs) * spot_prices * transaction_cost)
    running_pnl = np.cumsum(pnl, axis = 1)
    running_pnl_tcost = np.delete(running_pnl, -1, axis=1)
        
    return running_pnl_tcost

def delta_trigger(deltas, delta_threshold):
    """
    Checks whether the trigger based strategy is triggered according to the delta_threshold
    given by the user.

    Parameters
    ----------
    deltas : ndarray
        The delta values of the option at each time step
    delta_threshold : float
        The threshold that determines the trigger for delta-hedging

    Returns
    -------
    delta_changes : ndarray
        Contains the values where delta will be rehedged by if the difference is larger than the threshold.
    avg_rehedge_count : float
        The average value of rehedgings in each stock price path.

    """
    delta_changes = []
    rehedge_count = 0
    for i in range(len(deltas)):
        delta_index = []
        initial_delta = deltas[0,0]
        for j in range(len(deltas[i])):
            if abs(deltas[i,j] - initial_delta) < delta_threshold:
                delta_index.append(0)
            else:
                delta_index.append(abs(deltas[i,j] - initial_delta))
                initial_delta = deltas[i,j]
                rehedge_count += 1
        delta_changes.append(delta_index)
    delta_changes = np.array(delta_changes)
    avg_rehedge_count = rehedge_count / deltas.shape[0]
    
    return delta_changes, avg_rehedge_count

def gamma_delta_trigger(deltas, gammas, gamma_low, gamma_medium, delta_threshold_1, delta_threshold_2, delta_threshold_3):
    """
    Calculates the delta rehedgings based on the gamma levels and delta thresholds given by the user.
    
    Parameters
    ----------
    deltas : ndarray
        The delta values of the option at each time step
    gammas : ndarray
        The gamma values of the option at each time step
    gamma_low: float
        The lower gamma level provided by the user
    gamma_medium: float
        The medium gamma level provided by the user
    delta_threshold_1: 
        The threshold that determines the trigger for delta-hedging at a low gamma area.
    delta_threshold_2: 
        The threshold that determines the trigger for delta-hedging at a medium gamma area.
    delta_threshold_3: 
        The threshold that determines the trigger for delta-hedging at a high gamma area.
        
    Returns
    ------
    delta_changes : ndarray
        Contains the values where delta will be rehedged by if the difference is larger than the threshold.
    avg_rehedge_count : float
        The average value of rehedgings in each stock price path.
    """
    
    delta_changes = np.zeros_like(deltas)
    rehedge_count = 0
    for i in range(deltas.shape[0]):
        initial_delta = deltas[i,0]
        for j in range(gammas.shape[1]):
            if gammas[i,j] <= gamma_low:
                if abs(deltas[i,j] - initial_delta) >= delta_threshold_1:
                    delta_changes[i,j] = abs(deltas[i,j] - initial_delta)
                    initial_delta = deltas[i,j]
                    rehedge_count += 1
            elif gammas[i,j] <= gamma_medium:
                if abs(deltas[i,j] - initial_delta) >= delta_threshold_2:
                    delta_changes[i,j] = abs(deltas[i,j] - initial_delta)
                    initial_delta = deltas[i,j]
                    rehedge_count += 1
            else:
                if abs(deltas[i,j] - initial_delta) >= delta_threshold_3:
                    delta_changes[i,j] = abs(deltas[i,j] - initial_delta)
                    initial_delta = deltas[i,j]
                    rehedge_count +=1
    
    avg_rehedge_count = rehedge_count/deltas.shape[0]
    return delta_changes, avg_rehedge_count


def delta_hedge_simulator_no_tcost(S, K, r, T, steps, simulations, iv, rv):
    """
    Simulates the no-transaction cost delta hedging strategy and computes the PnL.

    Parameters
    ----------
    S : int
        Initial stock price
    K : int
        Strike price
    r : float
        Annual interest rate
    T : int
        The time to expiry(in years)
    steps : int
        Number of time steps in a time period
    simulations : int
        Number of simulated paths
    transaction_cost : float
        The cost of buying and selling the stock to rehedge the delta
    iv : float
        Implied volatility of the option
    rv : float
        Realized volatility of the option

    Returns
    -------
    running_pnl : ndarray
        The running PnL of the no-transaction cost delta-hedging strategy.

    """
    
    spot_prices = MonteCarloEngine(S, r, iv, T, steps, simulations)
        
    call_prices, deltas, gammas = calculate_call_price(spot_prices, K, r, iv, T, steps)
       
    running_pnl = compute_pnl_no_tcost(spot_prices, gammas, rv, iv, steps, T)
    
    return running_pnl
    
    
def delta_hedge_simulator(S, K, r, T, steps, simulations, transaction_cost, iv, rv, delta_threshold = None):
    """
    Simulates the trigger-based delta hedging strategy and computes the PnL

    Parameters
    ----------
    S : int
        Initial stock price
    K : int
        Strike price
    r : float
        Annual interest rate
    T : int
        The time to expiry(in years)
    steps : int
        Number of time steps in a time period
    simulations : int
        Number of simulated paths
    transaction_cost : float
        The cost of buying and selling the stock to rehedge the delta
    iv : float
        Implied volatility of the option
    rv : float
        Realized volatility of the option
    delta_threshold : float
        The threshold that determines the trigger for delta-hedging

    Returns
    -------
    running_pnl : ndarray
        The running PnL of the trigger-based delta-hedging strategy.
    avg_rehedge_count : float
        The average value of rehedgings in each stock price path.

    """
    
    spot_prices = MonteCarloEngine(S, r, iv, T, steps, simulations)
        
    call_prices, deltas, gammas = calculate_call_price(spot_prices, K, r, iv, T, steps)
    
    if delta_threshold is None:
        running_pnl = compute_pnl_tcost(spot_prices, deltas, gammas, rv, iv, steps, T, transaction_cost)
        avg_rehedge_count = steps

    else:
    
        delta_changes_trigger, avg_rehedge_count = delta_trigger(deltas, delta_threshold)
        
        running_pnl = compute_pnl_tcost(spot_prices, deltas, gammas, rv, iv, steps, T, transaction_cost,
                                        delta_changes = delta_changes_trigger)
    
    return running_pnl, avg_rehedge_count

def delta_gamma_hedge_simulator(S, K, r, T, steps, simulations, transaction_cost, iv, rv, gamma_low, gamma_medium,
                                delta_threshold_1, delta_threshold_2, delta_threshold_3):
    """
    Simulates the gamma-level based delta hedging strategy and computes the PnL

    Parameters
    ----------
    S : int
        Initial stock price
    K : int
        Strike price
    r : float
        Annual interest rate
    T : int
        The time to expiry(in years)
    steps : int
        Number of time steps in a time period
    simulations : int
        Number of simulated paths
    transaction_cost : float
        The cost of buying and selling the stock to rehedge the delta
    iv : float
        Implied volatility of the option
    rv : float
        Realized volatility of the option
    gamma_low: float
        The lower gamma level provided by the user
    gamma_medium: float
        The medium gamma level provided by the user
    delta_threshold1: 
        The threshold that determines the trigger for delta-hedging at a low gamma area.
    delta_threshold2: 
        The threshold that determines the trigger for delta-hedging at a medium gamma area.
    delta_threshold1: 
        The threshold that determines the trigger for delta-hedging at a high gamma area.

    Returns
    -------
    running_pnl : ndarray
        The running PnL of the trigger-based delta-hedging strategy.
    avg_rehedge_count : float
        The average value of rehedgings in each stock price path.
        
    """
    
    spot_prices = MonteCarloEngine(S, r, iv, T, steps, simulations)
        
    call_prices, deltas, gammas = calculate_call_price(spot_prices, K, r, iv, T, steps)
    
    
    delta_changes_trigger, avg_rehedge_count = gamma_delta_trigger(deltas, gammas, gamma_low, gamma_medium, delta_threshold_1, delta_threshold_2, delta_threshold_3)
        
    running_pnl = compute_pnl_tcost(spot_prices, deltas, gammas, rv, iv, steps, T, transaction_cost,
                                        delta_changes = delta_changes_trigger)
    
    return running_pnl, avg_rehedge_count


def delta_hedge_simulator_time_based(S, K, r, T, steps, simulations, transaction_cost, iv, rv, rehedges):
    """
    Simulates the time-based delta hedging strategy and computes the PnL

    Parameters
    ----------
    S : int
        Initial stock price
    K : int
        Strike price
    r : float
        Annual interest rate
    T : int
        The time to expiry(in years)
    steps : int
        Number of time steps in a time period
    simulations : int
        Number of simulated paths
    transaction_cost : float
        The cost of buying and selling the stock to rehedge the delta
    iv : float
        Implied volatility of the option
    rv : float
        Realized volatility of the option
    rehedges: int
        The number of steps between each rehedging operation.
        

    Returns
    -------
    running_pnl : ndarray
        The running PnL of the trigger-based delta-hedging strategy.

    """
    
    spot_prices = MonteCarloEngine(S, r, iv, T, steps, simulations)
        
    call_prices, deltas, gammas = calculate_call_price(spot_prices, K, r, iv, T, steps)
    
    running_pnl = compute_pnl_time_based(spot_prices, deltas, gammas, rv, iv, steps, T, transaction_cost, rehedges)
    
    return running_pnl


def LSV_volatility_correction(iv, transaction_cost, T, steps):
    """
    Adjusts the implied volatility according to Leland's formula.
    Parameters
    ----------
    iv : float
        Implied volatility of the option
    transaction_cost : float
        The cost of buying and selling the stock to rehedge the delta
    T : int
        The time to expiry(in years)
    steps : int
        Number of time steps in a time period

    Returns
    -------
    adjusted_implied_volatility : ndarray
        The implied volatility adjusted according to Leland's formula.

    """
    
    sqrt_2_over_pi_transaction_cost = transaction_cost * math.sqrt(2 / math.pi)
    adjusted_implied_volatility = []
    for i in range(steps):
        dt = T * (steps - i) / steps
        sqrt_iv_dt = iv * math.sqrt(dt)
        adjusted_iv = iv * ((1 + sqrt_2_over_pi_transaction_cost / sqrt_iv_dt)**0.5)
        adjusted_implied_volatility.append(adjusted_iv)
    return adjusted_implied_volatility

def LSV_volatility_correction2(iv, transaction_cost, T, steps):
    """
    Adjusts the implied volatility according to Leland's formula, Wilmott Approach.
    Parameters
    ----------
    iv : float
        Implied volatility of the option
    transaction_cost : float
        The cost of buying and selling the stock to rehedge the delta
    T : int
        The time to expiry(in years)
    steps : int
        Number of time steps in a time period

    Returns
    -------
    adjusted_implied_volatility : ndarray
        The implied volatility adjusted according to Wilmott's volatility correction formula.

    """
    
    sqrt_2_over_pi = math.sqrt(2 / math.pi)
    round_trip_cost = 2 * transaction_cost * iv
    adjusted_implied_volatility = []
    for i in range(steps):
        dt = T * (steps - i) / steps
        sqrt_dt = math.sqrt(1 / dt)
        adjusted_iv_square = iv ** 2 + (round_trip_cost * (sqrt_2_over_pi * sqrt_dt))
        adjusted_implied_volatility.append(math.sqrt(adjusted_iv_square))
    return adjusted_implied_volatility

def compute_pnl_leland(spot_prices, gammas, rv, leland_iv, steps, T):
    """
    Computes the running PnL of the delta-hedging strategy when Leland correction is used

    Parameters
    ----------
    spot_prices : ndarray
       Current prices of the underlying asset 
    gammas : ndarray
        The gamma values of the option at each time step
    rv : float
        The realized volatility of the underlying asset
    leland_iv : ndarray
        The Leland-adjusted implied volatility
    steps : int
        Number of time steps to use in the calculation
    T : float
        Time to expiration of the option

    Returns
    -------
    running_pnl_leland : ndarray
        The running PnL of the delta-hedging strategy

    """
    
    dt = np.full((steps * T), 1/(steps*T))
    dt[0] = 0
    pnl = 0.5 * (rv - np.array(leland_iv)) * spot_prices**2 * gammas * dt
    running_pnl = np.cumsum(pnl, axis = 1)
    running_pnl_leland = np.delete(running_pnl, -1, axis=1)
    
    return running_pnl_leland

def compute_pnl_leland2(spot_prices, gammas, iv, leland_volatility, steps, T):
    """
    Computes the running PnL of the delta-hedging strategy when Wilmott's volatility correction approach is used

    Parameters
    ----------
    spot_prices : ndarray
       Current prices of the underlying asset 
    gammas : ndarray
        The gamma values of the option at each time step
    iv : float
        The implied volatility of the underlying asset
    leland_volatility : ndarray
        The Leland-adjusted volatility
    steps : int
        Number of time steps to use in the calculation
    T : float
        Time to expiration of the option

    Returns
    -------
    running_pnl_leland : ndarray
        The running PnL of the delta-hedging strategy

    """
    
    dt = np.full((steps * T), 1/(steps*T))
    dt[0] = 0
    pnl = 0.5 * (iv - np.array(leland_volatility)) * spot_prices**2 * gammas * dt
    running_pnl = np.cumsum(pnl, axis = 1)
    running_pnl_leland = np.delete(running_pnl, -1, axis=1)
    
    return running_pnl_leland


def delta_hedge_simulator_leland(S, K, r, T, steps, simulations, iv, rv, transaction_cost):
    """
    Simulates the delta hedging strategy for Leland PnL

    Parameters
    ----------
    S : int
        Initial stock price
    K : int
        Strike price
    r : float
        Annual interest rate
    T : int
        The time to expiry(in years)
    steps : int
        Number of time steps in a time period
    simulations : int
        Number of simulated paths
    iv : float
        Implied volatility of the option
    rv : float
        Realized volatility of the option
    transaction_cost : float
        The cost of buying and selling the stock to rehedge the delta


    Returns
    -------
    running_pnl_leland : ndarray
        The running PnL of the Leland delta-hedging strategy.
    leland_iv: ndarray
        The implied volatility adjusted according to Leland's formula.
    """
    
    spot_prices = MonteCarloEngine(S, r, iv, T, steps, simulations)
        
    call_prices, deltas, gammas = calculate_call_price(spot_prices, K, r, iv, T, steps)
    
    leland_iv = LSV_volatility_correction(iv, transaction_cost, T, steps)
       
    running_pnl_leland = compute_pnl_leland(spot_prices, gammas, rv, leland_iv, steps, T)
    
    return running_pnl_leland, leland_iv

def delta_hedge_simulator_leland2(S, K, r, T, steps, simulations, iv, rv, transaction_cost):
    """
    Simulates the delta hedging strategy based on Wilmott's formula for Leland PnL

    Parameters
    ----------
    S : int
        Initial stock price
    K : int
        Strike price
    r : float
        Annual interest rate
    T : int
        The time to expiry(in years)
    steps : int
        Number of time steps in a time period
    simulations : int
        Number of simulated paths
    iv : float
        Implied volatility of the option
    rv : float
        Realized volatility of the option
    transaction_cost : float
        The cost of buying and selling the stock to rehedge the delta


    Returns
    -------
    running_pnl_leland : ndarray
        The running PnL when the Wilmott's volatility correction formula is applied.
    leland_volatility: ndarray
        The implied volatility adjusted according to Wilmott's volatility correction formula.
    """
    
    spot_prices = MonteCarloEngine(S, r, iv, T, steps, simulations)
        
    call_prices, deltas, gammas = calculate_call_price(spot_prices, K, r, iv, T, steps)
    
    leland_volatility = LSV_volatility_correction2(iv, transaction_cost, T, steps)
       
    running_pnl_leland = compute_pnl_leland2(spot_prices, gammas, iv, leland_volatility, steps, T)
    
    return running_pnl_leland, leland_volatility

