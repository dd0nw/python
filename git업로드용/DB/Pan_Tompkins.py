#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np  # numpy 임포트 추가
import pandas as pd

class Pan_Tompkins_QRS:
    def __init__(self, fs):
        self.fs = fs  # 샘플링 주파수를 클래스 속성으로 저장

    def band_pass_filter(self, signal):
        '''
        Band Pass Filter
        :param signal: input signal
        :return: processed signal
        '''
        # Initialize result
        result = None

        # Create a copy of the input signal
        sig = signal.copy()

        # Apply the low pass filter using the equation given
        for index in range(len(signal)):
            sig[index] = signal[index]

            if index >= 1:
                sig[index] += 2 * sig[index-1]

            if index >= 2:
                sig[index] -= sig[index-2]

            if index >= 6:
                sig[index] -= 2 * signal[index-6]

            if index >= 12:
                sig[index] += signal[index-12]

        # Copy the result of the low pass filter
        result = sig.copy()

        # Apply the high pass filter using the equation given
        for index in range(len(signal)):
            result[index] = -1 * sig[index]

            if index >= 1:
                result[index] -= result[index-1]

            if index >= 16:
                result[index] += 32 * sig[index-16]

            if index >= 32:
                result[index] += sig[index-32]

        # Normalize the result from the high pass filter
        max_val = max(max(result), -min(result))
        result = result / max_val

        return result

    def derivative(self, signal):
        result = signal.copy()

        # Apply the derivative filter using the equation given
        for index in range(len(signal)):
            result[index] = 0

            if index >= 1:
                result[index] -= 2 * signal[index-1]

            if index >= 2:
                result[index] -= signal[index-2]

            if index >= 2 and index <= len(signal) - 2:
                result[index] += 2 * signal[index+1]

            if index >= 2 and index <= len(signal) - 3:
                result[index] += signal[index+2]

            result[index] = (result[index] * self.fs) / 8  # self.fs 사용

        return result

    def squaring(self, signal):
        '''
        Squaring the Signal
        :param signal: input signal
        :return: processed signal
        '''
        # Initialize result
        result = signal.copy()

        # Apply the squaring using the equation given
        for index in range(len(signal)):
            result[index] = signal[index] ** 2

        return result

    def moving_window_integration(self, signal):
        '''
        Moving Window Integrator
        :param signal: input signal
        :return: processed signal
        '''
        # Initialize result and window size for integration
        result = signal.copy()
        win_size = round(0.150 * self.fs)  # self.fs 사용
        sum_val = 0

        # Calculate the sum for the first N terms
        for j in range(win_size):
            sum_val += signal[j] / win_size
            result[j] = sum_val

        # Apply the moving window integration using the equation given
        for index in range(win_size, len(signal)):
            sum_val += signal[index] / win_size
            sum_val -= signal[index-win_size] / win_size
            result[index] = sum_val

        return result

    def solve(self, signal):
        input_signal = signal.iloc[:, 1].to_numpy()

        # Bandpass Filter
        bpass = self.band_pass_filter(input_signal.copy())

        # Derivative Function
        der = self.derivative(bpass.copy())

        # Squaring Function
        sqr = self.squaring(der.copy())

        # Moving Window Integration Function
        mwin = self.moving_window_integration(sqr.copy())

        return mwin, bpass  # mwin과 bpass를 반환

