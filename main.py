import os
import sys
import cv2
import h5py
import numpy as np
import pandas as pd
import zwoasi as asi
from pathlib import Path
from collections import deque
from lmfit import minimize, Parameters
from datetime import datetime, timezone

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from PySide6.QtQuick import QQuickImageProvider
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject, Signal, Slot, QTimer, QUrl, QThread

class mockCamera():
    def __init__(self):
        self.camera_properties = {"MaxHeight": 1200, "MaxWidth": 6248}
        self.camera_controls = {'Gain': {'MinValue': 0, 'MaxValue': 700, 'ControlType': 0}, 'Exposure': {'MinValue': 0, 'MaxValue': 2000, 'ControlType': 1}, 'Temperature': {'ControlType': 8}, 'TargetTemp': {'MinValue': -10, 'MaxValue': 30, 'ControlType': 16}, 'CoolerOn': {'ControlType': 17}, 'AntiDewHeater': {'ControlType': 21}}
        self.height = 0
        self.width = 0
        self.exposure = 0
        self.gain = 0
        self.temperature = 0
        self.target_temp = 0
        self.t = 0
        self.temp_ramp = 0
        self.isCoolOn = False
        self.isHeaterOn = False

    def set_control_value(self, control_type, value):
        if control_type == 0:
            self.gain = value
        elif control_type == 1:
            self.exposure = value
        elif control_type == 16:
            self.target_temp = value
        elif control_type == 17:
            self.isCoolOn = value
        elif control_type == 21:
            self.isHeaterOn = value

    def get_control_value(self, control_type):
        if control_type == 0:
            return self.gain
        elif control_type == 1:
            return self.exposure
        elif control_type == 8:
            self.current_temperature()
            return (10*self.temperature, self.temperature)
        elif control_type == 16:
            return self.target_temp
        elif control_type == 17:
            return self.isCoolOn
        elif control_type == 21:
            return self.isHeaterOn

    def current_temperature(self):
        if self.temperature == self.target_temp:
            return

        temp_step = 1
        if(self.target_temp < self.temperature):
            self.temperature -= temp_step
        else:
            self.temperature += temp_step

    def get_camera_property(self):
        return self.camera_properties

    def get_controls(self):
        return self.camera_controls

    def set_roi_format(self, width, height, not_important, data_type):
        self.height = height
        self.width = width
        return

    def start_video_capture(self):
        return

    def get_video_data(self):
        return np.random.randint(0, 256, self.height * self.width).astype(np.uint16)

### All Internal Variables Length Scale Held at Millimeters (mm) For Ease of Calculations ###
### Outputted Variables Converted to Nanometers (nm) ###
### "The Optics of Spectroscopy" by J.M. Lerner and A. Thevenon is Frequently Cited and Will be Referred to as TOOS in Shorthand ###
class WavelengthCalibrator():

    def __init__(self):
        ### Initial Parameters According to ISA HR-320 Manual ###
        self.k = 1 # First Order Wavelength
        self.n = 1200 # Grating Period (g/mm)
        self.F = 320 # Focal Length of Spectrograph (mm)
        self.Dv = 24 * (np.pi/180) # Deviation Angle (Radians)
        self.gamma = 2.417 * (np.pi/180) # Rotation of grating relative to focal plane (Radians)
        self.width = 68 # Width of the Grating (mm)

        ### Initial Parameters According to ZWO Astro ASI2600 Pro Camera Documentation ###
        self.Pw = 0.00376 # Pitch of Detector --> The spacing between pixels (mm)
        self.P_Min = 0 # Pixel Number Corresponding to Minimum Wavelength
        self.P_Max = 6247 # Pixel Number Corresponding to Maximum Wavelength
        self.Pc = 3124 # Center Pixel Number

        ### Wavelength Parameters --> The Central Wavelength for Calibration is 633 nm and the Max-Min Wavelengths Will Be Computed Later ###
        self.lambda_c = 633 * 1e-6 # Central Wavelength of Spectrometer (mm)
        self.Wl_Min = None # The Minimum Wavelength Detectable by the Spectrometer (mm)
        self.Wl_Max = None # Maximum Wavelength Detectable by the Spectrometer (mm)

        ### The Reference Ne I spectra from NIST ###
        self.reference_spectra = None # Reference Ne I spectral line wavelength --> From NIST (nm) later converted to (mm)

        ### Other Relavent Variables Related to the NIST Reference Spectra ###
        self.transition_probability_threshold = 4.0 * 1e6 # Apparent minimum transition rate probability detectable by spectrometer (1/s)
        self.intensity_threshold = 5000 # Apparent minimum relative intensity detectable by spectrometer

    # Reset all variables to default manufacturer variables
    def Reset(self):
        self.Pc = 3124 # Central Pixel
        self.Dv = 24 * (np.pi/180) # Deviation Angle (Radians)
        self.gamma = 2.417 * (np.pi/180) # Rotation of grating relative to focal plane (Radians)
        self.lambda_c = 633 * 1e-6 # Central Wavelength (nm)

    def Process_Reference_Spectra(self, lambda_0, Wl_Min, Wl_Max, transition_rate_threshold, intensity_threshold, Wl_buffer = 2.0):
        # Filter the reference spectrum to only include the wavelengths between the theoretical minimum and maximum wavelengths (plus or minus a buffer) calculated by the calibrator algorithm with the initial parameters as provided by the spectrometer and camera parameters
        # Also, filter by the minimum transition rate threshold that can be determiend by the spectrometer (This is determined through trial and error so may not be completely accurate)
        Wl_Min -= Wl_buffer
        Wl_Max += Wl_buffer
        spectra = self.reference_spectra[(self.reference_spectra['obs_wl_air(mm)'] > Wl_Min) & (self.reference_spectra['obs_wl_air(mm)'] < Wl_Max) & ( (self.reference_spectra['Aki(s^-1)'] > transition_rate_threshold) | (self.reference_spectra['intens'] >= intensity_threshold) )]
        NIST_wavelengths = spectra['obs_wl_air(mm)'].to_numpy()

        # Sort the approximate wavelengths and NIST wavelengths to make matching possible through the Numpy searchsorted function
        lambda_0 = np.sort(lambda_0)
        NIST_wavelengths = np.sort(NIST_wavelengths)

        # The searchsorted function takes array A and B, it returns the indices of A which would maintain order of A if the elements of B were placed before these indices
        # The clip function just ensures there are no values which are outside the index range of array A. Any values outside this range are clipped to the bottom index or upper index (Depending on which is closer)
        i = np.searchsorted(NIST_wavelengths, lambda_0)
        i = np.clip(i, 1, len(NIST_wavelengths) - 1)

        # Search sorted returns the indices where elements of array B would be sorted if placed in array A. The indices i satisfy the relation: A[i-1] < B <= A[i]
        # Now we need to figure out which wavelength is closer, the left neighbor or the right neighbor
        # We subtract the search_sorted left neighbors (i-1) from the approximate wavelengths --> Call this C
        # Then we subtract the approximate wavelengths from the search_sorted right neighbors (i) --> Call this D
        # We then do a boolean check to see if C is smaller than D --> If smaller, Boolean evaluates to 1 otherwise it evaluates to 0
        # We then shift the indices i according to this boolean check. Shift left by 1 if C < D (Selects Left Neighbor) or remain at index i (Selects Right Neighbor)
        matched_wavelengths = NIST_wavelengths[i - (lambda_0 - NIST_wavelengths[i-1] < NIST_wavelengths[i] - lambda_0)]

        return matched_wavelengths

    # Compute a Gaussian Around the Detected Pixel
    def Gaussian(self, x, A, P0, sigma, B):
        return A*np.exp(-(x-P0)**2/(2*sigma**2)) + B

    def Extract_Peak_Centers(self, data, prominence = 0.0001):
        # Collapse 2D CCD data to 1D spectra by taking median of each column
        spectrum = np.mean(data, axis = 0)
        spectrum = spectrum/np.trapezoid(spectrum)

        # Smooth the Data
        spectrum = gaussian_filter1d(spectrum, sigma=1)

        # Determine the Peaks
        peaks, properties = find_peaks(spectrum, prominence = prominence, distance = 15)

        peak_centers = []

        # Compute a Gaussian Around the Peak
        # We find the Tippy-Top of the Peak, However Due to the Discrete Nature of the Pixel Detection this can Be Choppy
        # The Tippy-Top of the Peak may not be at the Center of the Line so we fit a Gaussian in the Region Around the Peak
        # We Then Take The Gaussian Center as the True Center
        for p in peaks:
            # Choose Window Around Peak
            lo = max(0, p-5)
            hi = min(len(spectrum), p+6)
            x = np.arange(lo, hi)
            y = spectrum[x]

            # Initial Guess for Parameters
            A0 = y.max() - y.min()
            P0 = p
            sigma0 = 1.5
            B0 = y.min()

            # Fit a Gaussian to The Peak
            popt, _ = curve_fit(self.Gaussian, x, y, p0=[A0, P0, sigma0, B0])
            peak_centers.append(popt[1])

        return spectrum, np.array(peak_centers)

    # To get an intensity array, we take the mean along the X-Axis (Recall that wavelengths only change along Y-Axis)
    # Then we divide by the trapezoidal integration of the intensity array to normalize the integration to be 1
    def Process_Pixel_Intensity(self, data):
        intensity = np.mean(data, axis = 0)
        intensity = intensity/np.trapezoid(intensity)
        return intensity

    # Utilizing Equations from "The Optics of Spectroscopy" by J.M. Lerner and A. Thevenon to Compute the Wavelength For a Given Pixel Detection
    def Get_Calibrated_Wavelength_Intensity_From_Pixel(self, data):
        # Compute the average, integration normalized intensity along the X-Axis
        intensity = self.Process_Pixel_Intensity(data)

        # Create an array for every single X-axis pixel
        pixels = np.arange(self.P_Min, self.P_Max + 1, 1)

        # Return the wavelengths (nm) and the intensity
        wavelength = self.Get_Wavelength_From_Pixel(self.k, self.n, self.F, self.Dv, self.gamma, self.Pw, self.Pc, self.lambda_c, pixels) * 1e6

        return  wavelength, intensity

    # Utilizing Equations from "The Optics of Spectroscopy" by J.M. Lerner and A. Thevenon to Compute the Wavelength For a Given Pixel Detection
    def Get_Wavelength_From_Pixel(self, k, n, F, Dv, gamma, Pw, Pc, lambda_c, P_lambda):
        alpha = self.Alpha(k, n, Dv, lambda_c)
        beta_lambda_c = self.Beta(k, n, alpha, lambda_c)
        beta_h = self.Beta_H(beta_lambda_c, gamma)
        Lh = self.LH(F, gamma)
        Hb_lambda_c = self.HB_Lambda_C(F, gamma)
        Hb_lambda_n = self.HB_Lambda_N_From_Pixel(Pw, P_lambda, Pc, Hb_lambda_c)
        beta_lambda_n = self.Beta_Lambda_N(beta_h, Hb_lambda_n, Lh)
        return self.Get_Wavelength(k, n, alpha, beta_lambda_n)

    # Utilizing Equations from "The Optics of Spectroscopy" by J.M. Lerner and A. Thevenon to Compute the Pixel Activated For a Given Wavelength
    def P_Lambda(self, k, n, Pw, Pc, F, Dv, gamma, lambda_c, lambda_n):
        alpha = self.Alpha(k, n, Dv, lambda_c)
        beta_lambda_c = self.Beta(k, n, alpha, lambda_c)
        beta_lambda_n = self.Beta(k, n, alpha, lambda_n)
        beta_h = self.Beta_H(beta_lambda_c, gamma)
        Lh = self.LH(F, gamma)
        Hb_lambda_c = self.HB_Lambda_C(F, gamma)
        Hb_lambda_n = self.HB_Lambda_N_From_Beta(Lh, beta_h, beta_lambda_n)
        return (Hb_lambda_c - Hb_lambda_n)/Pw + Pc

    # The Small Wavelength Difference Which Cannot be Resolved by the Diffraction Grating (Equation 5-11 in TOOS)
    def Dlambda(self, n, width, lambda_c):
        return lambda_c/(n*width)

    # The Angle for the Incoming Wavelength (Equation 2-1 in TOOS)
    def Alpha(self, k, n, Dv, lambda_n):
        return np.arcsin((k*n*lambda_n)/(2 * np.cos(Dv/2))) - Dv/2

    # The Angle for the Outgoing Wavelength (Equation 1-1 in TOOS)
    def Beta(self, k, n, alpha, lambda_n):
        return np.arcsin(k*n*lambda_n - np.sin(alpha))

    # The Angle from LH to the Normal to the Grating (Equation 5-4 in TOOS)
    def Beta_H(self, beta_lambda_c, gamma):
        return gamma + beta_lambda_c

    # The Outgoing Angle for Off-Center Wavelengths (Equation 5-7 in TOOS)
    def Beta_Lambda_N(self, beta_h, Hb_lambda_n, Lh):
        return beta_h - np.arctan(Hb_lambda_n/Lh)

    # Perpendicular Distance From Grating or Focusing Mirror to the Focal Plane (Equation 5-3 in TOOS)
    def LH(self, F, gamma):
        return F * np.cos(gamma)

    # Distance From the Intercept of the Normal to the Focal Plane to the Central Wavelength (Equation 5-5 in TOOS)
    def HB_Lambda_C(self, F, gamma):
        return F * np.sin(gamma)

    # Distance From the Intercept of the Normal to the Focal Plane to the Central Wavelength Computed From Outgoing Angle, Beta_Lambda_N (Equation 5-8 in TOOS)
    def HB_Lambda_N_From_Beta(self, Lh, beta_h, beta_lambda_n):
        return Lh * np.tan(beta_h - beta_lambda_n)

    # Distance From the Intercept of the Normal to the Focal Plane to the Central Wavelength Computed From Activated Pixel, P_Lambda (Equation 5-6 in TOOS)
    def HB_Lambda_N_From_Pixel(self, Pw, P_lambda, Pc, Hb_lambda_c):
        return Pw * (Pc - P_lambda) + Hb_lambda_c

    # Compute the Wavelength from the Incoming and Outgoing Angles of the Light Utilizing the Grating Equation (Equation 1-1 in TOOS)
    def Get_Wavelength(self, k, n, alpha, beta):
        return (np.sin(alpha) + np.sin(beta))/(k*n)

    # Get wavelength range in nanometers (nm)
    def Get_Wavelength_Range(self):
        self.Wl_Min = self.Get_Wavelength_From_Pixel(self.k, self.n, self.F, self.Dv, self.gamma, self.Pw, self.Pc, self.lambda_c, self.P_Min)
        self.Wl_Max = self.Get_Wavelength_From_Pixel(self.k, self.n, self.F, self.Dv, self.gamma, self.Pw, self.Pc, self.lambda_c, self.P_Max)
        return (self.Wl_Min*1e6, self.Wl_Max*1e6)

    # Get the central wavelength in nanometers (nm)
    def Get_Central_Wavelength(self):
        return self.lambda_c * 1e6

    def Get_Free_Parameters(self):
        return {'Pc': self.Pc, 'Dv': self.Dv, 'gamma': self.gamma, 'lambda_c': self.lambda_c}

    def Set_Free_Parameters(self, parameters):
        self.Pc = parameters['Pc']
        self.Dv = parameters['Dv']
        self.gamma = parameters['gamma']
        self.lambda_c = parameters['lambda_c']
        return

    # Compute the residual between the measured pixel position and the computed pixel position from the reference wavelength utilizing the current parameter combination
    def residual(self, params, P_measured, lambda_ref):
        # Unpack Parameters
        k = params['k']
        n = params['n']
        F = params['F']
        Pw = params['Pw']
        Pc = params['Pc']
        Dv = params['Dv']
        gamma = params['gamma']
        lambda_c = params['lambda_c']

        # Compute the Pixel Position for the Current Parameter Combination and Return the Residual with the Actual Pixel Positions Measured
        P_model = self.P_Lambda(k, n, Pw, Pc, F, Dv, gamma, lambda_c, lambda_ref)
        return (P_measured - P_model)

    # To Calibrate, we First Utilize a Reference He-Ne Laser with a Single Known Wavelength of 632.81646 nm
    # A Peak Finding Algorithm is Used to Find the Pixel Position of the He-Ne Line and Later the Lines for the Ne Spectrum
    # The Spectrometer was Dialed to a Center Wavelength of 633 nm so the He_Ne Line is Roughly at the Center Pixel Position
    # With the Center Pixel Position Determined, we then Estimate the Wavelengths of the Ne I Spectra Lines (Also centered to 633 nm when they were measured)
    # The Estimated Wavelengths are then Matched to the Closest Wavelengths in the NIST Reference Spectrum
    # The Center Pixel Position, the Deviation Angle, and the Tilt Angle are then Varied and Utilized to Compute Pixel Positions Using the Matched Reference Wavelengths
    # Residuals Between the Computed Pixel Position and the True Measured Pixel Positions are then Determined
    # A Least Squares Fit is Conducted to Minimize the Residuals and Find the Best Fit Parameter Combination Which Become the Calibrated Values
    def Calibrate(self, Ne_Data_Path, He_Ne_Line_Path, Nist_Reference_Path):
        # Preprocess the NIST Reference Spectra
        self.reference_spectra = pd.read_csv(Nist_Reference_Path)[['obs_wl_air(nm)', 'intens', 'Aki(s^-1)']].astype(float).dropna() # Reference Ne I spectral line wavelength --> From NIST (nm)
        self.reference_spectra['obs_wl_air(nm)'] *= 1e-6 # Convert wavelengths to millimeters for later calculations (mm)
        self.reference_spectra.rename(columns = {'obs_wl_air(nm)': 'obs_wl_air(mm)'}, inplace = True) # Rename column to reflect unit change

        # Flip the data along the X-axis since camera has the highest wavelengths on the left and the lowest on the right
        # We want the wavelengths to read low to high from left to right
        Ne_Data = np.load(Ne_Data_Path)
        Ne_Data = np.flip(Ne_Data, axis = 1)

        He_Ne_Line = np.load(He_Ne_Line_Path)
        He_Ne_Line = np.flip(He_Ne_Line, axis = 1)

        # Get Pixel Dimensions
        l,w = Ne_Data.shape

        # The He_Ne Reference Line has a single line at 632.81646 nm.
        # The Czerny-Turner Spectrometer was centered at 633 nm and we will utilize the He_Ne line as a reference line to determine central pixel position
        spectrum, peaks = self.Extract_Peak_Centers(He_Ne_Line)
        self.Pc = peaks[0]

        # Extract the Peak Centers from the measured Ne Spectrum
        spectrum, P_measured = self.Extract_Peak_Centers(Ne_Data)

        # Compute the minimum wavelength that can be resolved from our diffraction grating and ensure the residuals are smaller than it. Set initial residuals to infinity
        # If they are, calibration is complete. Otherwise, if 5 rounds of calibration pass without success, we consider the calibration a failure.
        d_lambda = self.Dlambda(self.n, self.width, self.lambda_c)
        residuals = np.ones(len(P_measured)) * np.inf
        iter = 0
        while(np.any(residuals > d_lambda) and iter < 5):
            # Compute the Minimum and Maximum Wavelengths With the Current Parameters (Default Manufacturer Parameters with Corrected Central Pixel Position)
            self.Wl_Min = self.Get_Wavelength_From_Pixel(self.k, self.n, self.F, self.Dv, self.gamma, self.Pw, self.Pc, self.lambda_c, self.P_Min)
            self.Wl_Max = self.Get_Wavelength_From_Pixel(self.k, self.n, self.F, self.Dv, self.gamma, self.Pw, self.Pc, self.lambda_c, self.P_Max)

            # Compute the Wavelengths for the Measured Pixels With the Current Parameters (Default Manufacturer Parameters with Corrected Central Pixel Position)
            lambda_0 = self.Get_Wavelength_From_Pixel(self.k, self.n, self.F, self.Dv, self.gamma, self.Pw, self.Pc, self.lambda_c, P_measured)

            # Utilizing the Minimum and Maximum Wavelength as Bounds, Find the Best Fit NIST Reference Lines to the Current Computed Wavelengths
            lambda_ref = self.Process_Reference_Spectra(lambda_0, self.Wl_Min, self.Wl_Max, self.transition_probability_threshold, self.intensity_threshold)

            # Initialize the Parameters to be Utilized in the Least Squares Fit
            # The Parameters to be Fit are the Central Pixel Position, the Deviation Angle, and the Tilt Angle as these are the most likely to be deviated from stated values
            params = Parameters()
            params.add('k', value = self.k, vary = False)
            params.add('n', value = self.n, vary = False)
            params.add('F', value = self.F, vary = False)
            params.add('Pw', value = self.Pw, vary = False)
            params.add('lambda_c', value = self.lambda_c)
            params.add('Pc', value = self.Pc, vary = True, min = self.Pc - self.Pc*0.1, max = self.Pc + self.Pc*0.1)
            params.add('Dv', value = self.Dv, vary = True, min = self.Dv - self.Dv*0.2, max = self.Dv + self.Dv*0.2)
            params.add('gamma', value = self.gamma, vary = False, min = self.gamma - self.gamma*0.3, max = self.gamma + self.gamma*0.3)

            # Compute the Fit by Miniimizing the Square of the Residual
            least_squares_fitter = minimize(self.residual, params, args=(P_measured, lambda_ref), method = "least_squares")

            # Compute the Wavelengths from the Measured Pixel Positions with the Calibrated Parameters
            calibrated_wavelengths = self.Get_Wavelength_From_Pixel(least_squares_fitter.params['k'], least_squares_fitter.params['n'], least_squares_fitter.params['F'], least_squares_fitter.params['Dv'], least_squares_fitter.params['gamma'], least_squares_fitter.params['Pw'], least_squares_fitter.params['Pc'], least_squares_fitter.params['lambda_c'], P_measured)

            # Set the Internal Parameters to the Calibrated Values
            self.Dv = float(least_squares_fitter.params['Dv'])
            self.gamma = float(least_squares_fitter.params['gamma'])
            self.Pc = int(least_squares_fitter.params['Pc'])

            # Compute the minimum wavelength that can be resolved from our diffraction grating
            d_lambda = self.Dlambda(self.n, self.width, self.lambda_c)

            # Compute the residual between the reference wavelength and the calibrated wavelength
            residuals = np.abs(lambda_ref - calibrated_wavelengths)

            # Update the iteration count
            iter += 1
        return

    # Once the parameters have been calibrated, the computation of wavelengths should be extremely close to the true wavelengths plus/minus an offset
    # This offset is due to the central wavelength on the spectrometer dial not being completely exact
    # We therefore shift the central wavelength while ensuring we fit to Ne I reference lines which ensures we correct for the offset
    # Once the parameters have been calibrated, the computation of wavelengths should be extremely close to the true wavelengths plus/minus an offset
    # This offset is due to the central wavelength on the spectrometer dial not being completely exact
    # We therefore shift the central wavelength while ensuring we fit to Ne I reference lines which ensures we correct for the offset
    def Central_Wavelength_Shift(self, Ne_Spectra_Path, lambda_c):
        # Set a new central wavelength
        self.lambda_c = lambda_c*1e-6

        # Read in the data and flip it to the correct orientation (Lower wavelengths on the left and higher wavelengths on the right)
        data = np.load(Ne_Spectra_Path)
        data = np.flip(data, axis = 1)

        # Get the peaks for the current Ne Spectra
        spectrum, P_measured = self.Extract_Peak_Centers(data)

        # Compute the minimum wavelength that can be resolved from our diffraction grating and ensure the residuals are smaller than it. Set initial residuals to infinity
        # If they are, calibration is complete. Otherwise, if 5 rounds of calibration pass without success, we consider the calibration a failure.
        d_lambda = self.Dlambda(self.n, self.width, self.lambda_c)
        residuals = np.ones(len(P_measured)) * np.inf
        iter = 0
        while(np.any(residuals > d_lambda) and iter < 5):
            # Get the minimum and maximum wavelength for the current central wavelength
            self.Wl_Min = self.Get_Wavelength_From_Pixel(self.k, self.n, self.F, self.Dv, self.gamma, self.Pw, self.Pc, self.lambda_c, self.P_Min)
            self.Wl_Max = self.Get_Wavelength_From_Pixel(self.k, self.n, self.F, self.Dv, self.gamma, self.Pw, self.Pc, self.lambda_c, self.P_Max)

            # Compute the wavelengths from the current calibrated parameters
            lambda_0 = self.Get_Wavelength_From_Pixel(self.k, self.n, self.F, self.Dv, self.gamma, self.Pw, self.Pc, self.lambda_c, P_measured)

            # Find the closest fitting NIST reference wavelengths to the computed wavelengths
            lambda_ref = self.Process_Reference_Spectra(lambda_0, self.Wl_Min, self.Wl_Max, self.transition_probability_threshold, self.intensity_threshold)

            # Create a parameter dictionary for our spectrometer/camera parameters to pass into the least squares fitting algorithm
            # We only vary the central wavelength, lambda_c, to correct the offset
            params = Parameters()
            params.add('k', value = self.k, vary = False)
            params.add('n', value = self.n, vary = False)
            params.add('F', value = self.F, vary = False)
            params.add('Pw', value = self.Pw, vary = False)
            params.add('Pc', value = self.Pc, vary = False)
            params.add('Dv', value = self.Dv, vary = False)
            params.add('gamma', value = self.gamma, vary = False)
            params.add('lambda_c', value = self.lambda_c, vary = True, min = self.lambda_c - self.lambda_c*0.1, max = self.lambda_c + self.lambda_c*0.1)

            # We minimize the square of the residual to determine the best fit central wavelength
            least_squares_fitter = minimize(self.residual, params, args=(P_measured, lambda_ref), method = 'least_squares')

            # Compute the Wavelengths from the Measured Pixel Positions with the Calibrated Parameters
            calibrated_wavelengths = self.Get_Wavelength_From_Pixel(least_squares_fitter.params['k'], least_squares_fitter.params['n'], least_squares_fitter.params['F'], least_squares_fitter.params['Dv'], least_squares_fitter.params['gamma'], least_squares_fitter.params['Pw'], least_squares_fitter.params['Pc'], least_squares_fitter.params['lambda_c'], P_measured)

            # Set the internal central wavelength to the calibrated central wavelength
            self.lambda_c = float(least_squares_fitter.params['lambda_c'])

            # Compute the minimum wavelength that can be resolved from our diffraction grating
            d_lambda = self.Dlambda(self.n, self.width, self.lambda_c)

            # Compute the residual between the reference wavelength and the calibrated wavelength
            residuals = np.abs(lambda_ref - calibrated_wavelengths)

            # Update the iteration count
            iter += 1

        # Check if calibration was complete or not by checking if the residuals are smaller than the resolution of the diffraction grating
        if(np.all(residuals < d_lambda)):
            return True, np.max(residuals)*1e6
        else:
            return False, np.max(residuals)*1e6

class CameraImageProvider(QQuickImageProvider):
    def __init__(self):
        super().__init__(QQuickImageProvider.Image)
        self.image = QImage()

    def requestImage(self, id, size, requestedSize):
        # Qt expects just a QImage here
        return self.image

    def updateImage(self, img: QImage):
        self.image = img

class CameraWorker(QObject):
    frameReady = Signal(QImage)  # for display
    standardCaptureFinished = Signal()
    captureBackgroundFinished = Signal()
    finished = Signal()

    def __init__(self, cam, max_width, max_height):
        super().__init__()
        self.cam = cam
        self.max_width = max_width
        self.max_height = max_height
        self.gain = None
        self.exposure = None

        self.isCooler = False
        self.isAntiDewHeater = False
        self.temp_c = None
        self.target_temp = None

        self.canCaptureSnapshot = False
        self.snapshot = None

        self.background = None
        self.subtraction_enabled = False

        self.background_n_frames_start = 0
        self.background_n_frames_end = 0
        self.background_n_frames_averaged = 0
        self.can_capture_background = False

        self.save_raw_enabled = False
        self.save_wavelength_enabled = False

        self.standard_capture_path = None
        self.standard_capture_n_frames_start = 0
        self.standard_capture_n_frames_end = 0
        self.can_standard_capture = False

        self.live_capture_running = False
        self.live_capture_path = None

        self.h5file = None
        self.h5_dataset_raw = None
        self.h5_dataset_wl = None
        self.h5_dataset_intens = None

        self.raw_data_str = "raw_data"
        self.wavelength_data_str = "wavelength"
        self.intensity_data_str = "intensity"

        self.save_queue = deque(maxlen=256)  # raw ring buffer

        self.wavelength_calibrator = WavelengthCalibrator()
        self.isCalibrated = False
        self.Pc = None
        self.Dv = None
        self.gamma = None
        self.lambda_c = None
        self.max_residual = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.display)

        self.save_timer = QTimer(self)
        self.save_timer.timeout.connect(self.flush_save_queue)

    def start(self):
        self.timer.start(500)
        self.save_timer.start(2000)  # write at 20 Hz
        #self.timer.start(int(exposure_ms * 1.1))

    def flush_save_queue(self):
        if not self.live_capture_running and not self.can_standard_capture:
            return

        if self.live_capture_running and self.can_standard_capture:
            return

        if not self.save_queue:
            return

        # Get the raw data from the queue and then clear the save queue
        raw_data_frames = list(self.save_queue)
        n = len(raw_data_frames)
        self.save_queue.clear()

        # Add the raw data to the h5 file if saving raw data is enabled by user
        if self.save_raw_enabled:
            self.h5_dataset_raw.resize(self.h5_dataset_raw.shape[0] + n, axis=0)
            self.h5_dataset_raw[-n:] = np.array(raw_data_frames)

        # Add the wavelength data to the h5 file if saving wavelength data is enabled by user
        if self.save_wavelength_enabled:
            # Compute the calibrated wavelength for each frame
            wavelengths = []
            intensities = []
            for frame in raw_data_frames:
                this_wavelength, this_intensity = self.wavelength_calibrator.Get_Calibrated_Wavelength_Intensity_From_Pixel(frame)
                wavelengths.append(this_wavelength)
                intensities.append(this_intensity)

            self.h5_dataset_wl.resize(self.h5_dataset_wl.shape[0] + n, axis=0)
            self.h5_dataset_wl[-n:] = np.array(wavelengths)

            self.h5_dataset_intens.resize(self.h5_dataset_intens.shape[0] + n, axis=0)
            self.h5_dataset_intens[-n:] = np.array(intensities)

    def display(self):
        frame = self.acquire_frame()

        if self.canCaptureSnapshot:
            self.snapshot = frame.copy()
            self.canCaptureSnapshot = False

        if self.can_capture_background:

            if self.background_n_frames_start == self.background_n_frames_end:
                self.background = self.background / self.background_n_frames_end
                self.background_n_frames_averaged = self.background_n_frames_end

                self.can_capture_background = False
                self.background_n_frames_start = 0
                self.background_n_frames_end = 0

                self.captureBackgroundFinished.emit()
            else:
                self.background = frame if self.background is None else self.background + frame
                self.background_n_frames_start += 1

        if self.background is not None and self.subtraction_enabled and self.background_n_frames_end == 0:
            frame = frame - self.background
            frame[frame < 0] = 0

        if self.can_standard_capture:
            if self.standard_capture_n_frames_start == self.standard_capture_n_frames_end:
                self.end_standard_capture()
            else:
                self.save_queue.append(frame.copy())
                self.standard_capture_n_frames_start += 1

        # Enqueue frame for live capture (NON-BLOCKING)
        if self.live_capture_running:
            self.save_queue.append(frame.copy())

        qimg = self.convert_to_QImage(frame)
        self.frameReady.emit(qimg)

    def acquire_frame(self):
        raw_data = self.cam.get_video_data()
        frame = np.frombuffer(raw_data, dtype=np.uint16).reshape(self.max_height, self.max_width)
        return frame

    def convert_to_QImage(self, frame):
        # Q-Image expects an 8 bit display so we convert to from 16-bit to 8-bit
        # We do this by min-max normalizing and then multiplying by 255 (Max value 8-bits can express)
        # Then convert to QImage for QML
        frame_8_bit = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return QImage(frame_8_bit.data, self.max_width, self.max_height, frame_8_bit.strides[0], QImage.Format_Grayscale8).copy()

    @Slot(int, float, float, float)
    def set_wavelength_calibration_variables(self, Pc, Dv, gamma, lambda_c):
        self.max_residual = None
        self.isCalibrated = True

        self.Pc = Pc
        self.Dv = Dv
        self.gamma = gamma
        self.lambda_c = lambda_c
        self.wavelength_calibrator.Set_Free_Parameters({'Pc': self.Pc, 'Dv': self.Dv, 'gamma': self.gamma, 'lambda_c': self.lambda_c})

    @Slot()
    def capture_snapshot(self):
        self.canCaptureSnapshot = True

    @Slot(str)
    def save_snapshot(self, file_path):
        np.save(file_path, self.snapshot)

    @Slot(int)
    def capture_background(self, n_frames):
        self.background = None
        self.can_capture_background = True
        self.background_n_frames_start = 0
        self.background_n_frames_end = n_frames

    @Slot(str)
    def load_background(self, path):
        if path.endswith(".npy"):
            self.background = np.load(path)
        elif path.endswith(".h5") or path.endswith(".hdf5"):
            with h5py.File(path, "r") as f:
                if "background" not in f:
                    raise KeyError("HDF5 file does not contain 'background' dataset")
                self.background = f["background"][()]
        else:
            raise ValueError("Unsupported background file format")

        # Safety checks
        if self.background.shape != (self.max_height, self.max_width):
            raise ValueError( f"Background shape mismatch: {self.background.shape}, "f"expected {(self.max_height, self.max_width)}")

        self.background_n_frames_averaged = 0
        return

    @Slot(bool)
    def enable_subtraction(self, enabled):
        self.subtraction_enabled = enabled

    @Slot(str)
    def save_background(self, path):
        if self.background is not None:
            np.save(path, self.background)

    @Slot(bool)
    def set_can_save_raw(self, status):
        self.save_raw_enabled = status

    @Slot(bool)
    def set_can_save_wavelength(self, status):
        self.save_wavelength_enabled = status

    def adjust_h5_file_name(self, path, file_prefix):
        prefix_adjusted = file_prefix + '_'
        old_path = Path(path)
        new_path = old_path.with_name(prefix_adjusted + old_path.name)
        return new_path

    def open_h5_file(self, path):
        # Open hdf5 file
        self.h5file = h5py.File(path, "w")

        # Set attributes relavent to current experiment
        ### Camera Attributes ###
        self.h5file.attrs["bit_depth"] = 16
        self.h5file.attrs["camera_width"] = self.max_width
        self.h5file.attrs["camera_height"] = self.max_height
        self.h5file.attrs["gain"] = self.gain
        self.h5file.attrs["exposure (ms)"] = self.exposure

        ### Temperature Attributes ###
        self.h5file.attrs["cooler_on"] = self.isCooler
        self.h5file.attrs["anti_dew_heater_on"] = self.isAntiDewHeater
        self.h5file.attrs["camera_target_temperature (celsius)"] = self.target_temp
        self.h5file.attrs["camera_start_temperature (celsius)"] = self.temp_c

        ### Background Attributes ###
        self.h5file.attrs["background_subtraction_applied"] = self.subtraction_enabled
        if self.subtraction_enabled:
            if self.background_n_frames_averaged == 0:
                self.h5file.attrs["background_source"] = "loaded"
            else:
                self.h5file.attrs["background_source"] = "captured"
                self.h5file.attrs["background_frames_averaged"] = self.background_n_frames_averaged

        ### Calibration Attributes ###
        if self.save_wavelength_enabled:
            self.h5file.attrs["wavelength_calibration_applied"] = self.isCalibrated
            if self.isCalibrated:
                self.h5file.attrs["central_pixel"] = self.Pc
                self.h5file.attrs["deviation_angle (rads)"] = self.Dv
                self.h5file.attrs["tilt_angle (rads)"] = self.gamma
                self.h5file.attrs["central_wavelength (nm)"] = self.lambda_c

                self.h5file.attrs["deviation_angle (degrees)"] = self.Dv * 180/np.pi
                self.h5file.attrs["tilt_angle (degrees)"] = self.gamma * 180/np.pi

                if self.max_residual is None:
                    self.h5file.attrs["wavelength_calibration_source"] = "loaded"
                else:
                    self.h5file.attrs["wavelength_calibration_source"] = "computed"
                    self.h5file.attrs["wavelength_calibration_max_residual"] = self.max_residual

        ### Capture Type Attributes ###
        self.h5file.attrs["acquisition_complete"] = False

        if self.live_capture_running:
            self.h5file.attrs["capture_type"] = "live"
        elif self.can_standard_capture:
            self.h5file.attrs["capture_type"] = "standard"

        ### Date-Time Attributes ###
        self.h5file.attrs["experiment_start_coordinated_universal_time"] = datetime.now(timezone.utc).isoformat()
        self.h5file.attrs["experiment_start_local_time"] = datetime.now().astimezone().isoformat()
        self.h5file.attrs["local_timezone"] = datetime.now().astimezone().tzname()

        # Create HDF5 file datasets
        # Save raw data if raw data is enabled, save wavelength and intensity if wavelength is enabled, save background if background subtraction is enabled
        if self.save_raw_enabled:
            self.h5_dataset_raw = self.h5file.create_dataset(self.raw_data_str, shape=(0, self.max_height, self.max_width), maxshape=(None, self.max_height, self.max_width), dtype=np.uint16, chunks=(1, self.max_height, self.max_width))
        if self.save_wavelength_enabled:
            self.h5_dataset_wl = self.h5file.create_dataset(self.wavelength_data_str, shape=(0, self.max_width), maxshape=(None, self.max_width), dtype=np.float64, chunks=(1, self.max_width))
            self.h5_dataset_intens = self.h5file.create_dataset(self.intensity_data_str, shape=(0, self.max_width), maxshape=(None, self.max_width), dtype=np.float64, chunks=(1, self.max_width))
        if self.subtraction_enabled and self.background is not None:
            self.h5_dataset_background = self.h5file.create_dataset("background", data=self.background, dtype=self.background.dtype, shape = (self.max_height, self.max_width))
        return

    def close_h5_file(self):
        # Ensure file exists
        if self.h5file is not None:

            # Add attributes upon closure
            ### Number of Frames Captured ###
            if self.h5_dataset_raw is not None:
                self.h5file.attrs["n_frames"] = self.h5_dataset_raw.shape[0]
            elif self.h5_dataset_wl is not None:
                self.h5file.attrs["n_frames"] = self.h5_dataset_wl.shape[0]

            ### Temperature attributes ###
            self.h5file.attrs["camera_end_temperature (celsius)"] = self.temp_c

             ### Date-Time Attributes ###
            self.h5file.attrs["experiment_end_coordinated_universal_time"] = datetime.now(timezone.utc).isoformat()
            self.h5file.attrs["experiment_end_local_time"] = datetime.now().astimezone().isoformat()

            # Compute Duration in Seconds of the Experiment
            start_iso = self.h5file.attrs["experiment_start_coordinated_universal_time"]
            end_iso   = self.h5file.attrs["experiment_end_coordinated_universal_time"]
            start_dt = datetime.fromisoformat(start_iso)
            end_dt   = datetime.fromisoformat(end_iso)

            ### Duration Attribute ###
            self.h5file.attrs["duration (s)"] = (end_dt - start_dt).total_seconds()

            ### Acquisition Completion Attribute ###
            self.h5file.attrs["acquisition_complete"] = True

            self.h5file.close()
            self.h5file = None
            self.h5_dataset_raw = None
            self.h5_dataset_wl = None

    @Slot(str)
    def set_standard_capture_save_path(self, path):
        self.standard_capture_path = path

    @Slot(str)
    def set_live_capture_save_path(self, path):
        self.live_capture_path = path

    @Slot(int)
    def start_standard_capture(self, n_frames):
        # Check that the user has chosen to save either raw or wavelength
        if self.standard_capture_path is not None:
            if self.save_raw_enabled or self.save_wavelength_enabled:
                self.can_standard_capture = True
                self.standard_capture_n_frames_start = 0
                self.standard_capture_n_frames_end = n_frames
                self.open_h5_file(self.standard_capture_path)
        return

    def end_standard_capture(self):
        self.flush_save_queue()
        self.can_standard_capture = False
        self.standard_capture_n_frames_start = 0
        self.standard_capture_n_frames_end = 0
        self.close_h5_file()
        self.standardCaptureFinished.emit()
        return

    @Slot()
    def start_live_capture(self):
        if self.live_capture_path is not None:
            # Check that the user has chosen to save either raw or wavelength
            if self.save_raw_enabled or self.save_wavelength_enabled:
                self.live_capture_running = True
                self.open_h5_file(self.live_capture_path)
        return

    @Slot()
    def end_live_capture(self):
        self.flush_save_queue()
        self.live_capture_running = False
        self.close_h5_file()
        return

    @Slot(int)
    def get_gain(self, gain):
        self.gain = gain

    @Slot(int)
    def get_exposure(self, exposure):
        self.exposure = exposure

    @Slot(int)
    def get_target_temp(self, target_temp):
        self.target_temp = target_temp

    @Slot(float)
    def get_current_temp(self, temp_c):
        self.temp_c = temp_c

    @Slot(float)
    def get_max_residual(self, max_residual):
        self.max_residual = max_residual

    @Slot(bool)
    def get_cooler_status(self, cooler_status):
        self.isCooler = cooler_status

    @Slot(bool)
    def get_antiDewHeater_status(self, heater_status):
        self.isAntiDewHeater = heater_status

class CameraController(QObject):
    # User Interface Signals
    frameReady = Signal(QImage)
    gainRangeChanged = Signal(int, int)
    exposureRangeChanged = Signal(int, int)
    tempRangeChanged = Signal(int, int)
    temperatureChanged = Signal(float)
    residualCalculated = Signal(float)
    canSaveBackgroundChanged = Signal(bool)
    canSubtractBackgroundChanged = Signal(bool)
    canResetBackgroundChanged = Signal(bool)
    canSaveCalibrationChanged = Signal(bool)
    centralWavelengthChanged = Signal(float)
    wavelengthRangeChanged = Signal(float, float)
    canStandardCaptureChanged = Signal(bool)
    canStartLiveCaptureChanged = Signal(bool)
    canEndLiveCaptureChanged = Signal(bool)
    canEditLiveCaptureFileChanged = Signal(bool)
    canEnableWavelengthCheckBox = Signal(bool)
    canEditControls = Signal(bool)
    notApplicableResidual = Signal()
    canCalibrate = Signal(bool)
    isCalibrated = Signal(str)
    liveCaptureStarted = Signal()
    liveCaptureStopped = Signal()
    errorOccurred = Signal(str)

    # Worker Signals
    calibrationVariablesRequested = Signal(int, float, float, float)
    gainRequested = Signal(int)
    exposureRequested = Signal(int)
    targetTempRequested = Signal(int)
    currentTempRequested = Signal(float)
    maxResidualRequested = Signal(float)
    coolerStatusRequested = Signal(bool)
    antiDewHeaterStatusRequested = Signal(bool)
    captureBackgroundRequested = Signal(int)
    standardCaptureRequested = Signal(int)
    saveBackgroundRequested = Signal(str)
    loadBackgroundRequested = Signal(str)
    enableBackgroundSubtractionRequested = Signal(bool)
    standardCaptureSavePathRequested = Signal(str)
    liveCaptureSavePathRequested = Signal(str)
    startLiveCaptureRequested = Signal()
    endLiveCaptureRequested = Signal()
    canSaveRawRequested = Signal(bool)
    canSaveWavelengthRequested = Signal(bool)
    snapshotRequested = Signal()
    saveSnapshotRequested = Signal(str)

    def __init__(self):
        super().__init__()
        # Path to the SDK library
        sdk_path = "C:/Users/dm223/OneDrive/Documents/Wagner_Docs/Scripts/Dependencies/ASI_Camera_SDK/ASI_Windows_SDK_V1.39/ASI SDK/lib/x64/ASICamera2.dll"

        # Init ASI
        asi.init(sdk_path)
        try:
            self.cam = asi.Camera(0)
        except:
            self.cam = mockCamera()
        info = self.cam.get_camera_property()
        self.max_height = info['MaxHeight']
        self.max_width = info['MaxWidth']
        self.cam.set_roi_format(self.max_width, self.max_height, 1, asi.ASI_IMG_RAW16)

        # Camera Worker
        self.camera_thread = None
        self.camera_worker = None

        # Data Acquisition
        self.is_live_capture_running = False
        self.is_standard_capture_running = False
        self.save_raw_enabled = False
        self.save_wavelength_enabled = False

        # Camera Controls
        controls = self.cam.get_controls()
        self.exposure_min = controls['Exposure']['MinValue']//1000
        self.exposure_max = min(controls['Exposure']['MaxValue']//1000, 5000)
        self.gain_min = max(controls['Gain']['MinValue'], 0)
        self.gain_max = controls['Gain']['MaxValue']

        # Heater/Cooler Controls:
        self.temp_min = controls['TargetTemp']['MinValue']
        self.temp_max = controls['TargetTemp']['MaxValue']

        # N Frames for Background / Data Acquisition
        self.is_capturing_background = False
        self.background_n_frames = 1
        self.standard_capture_n_frames = 1

        self.control_types = dict()
        for control in controls.keys():
            self.control_types[control] = controls[control]['ControlType']

        self.cam.start_video_capture()

        # Calibration
        self.wavelength_calibrator = WavelengthCalibrator()
        self.calibrationStatus = False
        self.calibration_save_enabled = False # Only true after a calibration has been initiated
        self.isCalibrationReadyToUse = False

        self.Nist_Reference_Path = None
        self.He_Ne_Path = None
        self.Ne_633_Anchor_Path = None
        self.Ne_Calibration_Path = None

        self.min_wavelength = 400
        self.approximate_central_wavelength = 700
        self.max_wavelength = 1000
        self.max_residual = None

        # Access cache to use previous calibrations
        if os.path.exists("cache/calibration.csv"):
            self.wavelength_calibrator.set_free_parameters(pd.read_csv("cache/calibration.csv"))

        # Another timer just for temperature updates
        self.temp_timer = QTimer()
        self.temp_timer.timeout.connect(self.update_temperature)
        self.temp_timer.start(1000)  # every 1 second

    @Slot()
    def initialize_controls(self):
        self.exposureRangeChanged.emit(self.exposure_min, self.exposure_max)
        self.gainRangeChanged.emit(self.gain_min, self.gain_max)
        self.tempRangeChanged.emit(self.temp_min, self.temp_max)

    @Slot(int)
    def setNBackgroundFrames(self, n):
        self.background_n_frames = n

    @Slot(int)
    def setNStandardCaptureFrames(self, n):
        self.standard_capture_n_frames = n

    @Slot(int)
    def setExposure(self, ms):
        self.cam.set_control_value(self.control_types['Exposure'], max(1000, ms*1000))
        self.exposureRequested.emit(ms)

    @Slot(int)
    def setGain(self, gain):
        self.cam.set_control_value(self.control_types['Gain'], gain)
        self.gainRequested.emit(gain)

    @Slot(int)
    def setTargetTemp(self, temp):
        self.cam.set_control_value(self.control_types['TargetTemp'], temp)
        self.targetTempRequested.emit(temp)

    @Slot()
    def update_temperature(self):
        try:
            temp_val = self.cam.get_control_value(self.control_types['Temperature'])[0]
            temp_c = temp_val // 10 # convert to °C
            self.temperatureChanged.emit(temp_c)
            self.currentTempRequested.emit(temp_c)
        except Exception as e:
            self.errorOccurred.emit("Error reading temperature: ", e)

    @Slot(bool)
    def setCooler(self, enabled: bool):
        try:
            self.cam.set_control_value(self.control_types['CoolerOn'], 1 if enabled else 0)
            self.coolerStatusRequested.emit(enabled)
        except Exception as e:
            self.errorOccurred.emit("Error setting cooler: ", e)

    @Slot(bool)
    def setAntiDewHeater(self, enabled: bool):
        try:
            self.cam.set_control_value(self.control_types['AntiDewHeater'], 1 if enabled else 0)
            self.antiDewHeaterStatusRequested.emit(enabled)
        except Exception as e:
            self.errorOccurred.emit("Error setting anti-dew heater: ", e)

    def isCalibrationReady(self):
        return (self.Nist_Reference_Path is not None) and (self.He_Ne_Path is not None) and (self.Ne_633_Anchor_Path is not None) and (self.Ne_Calibration_Path is not None)

    @Slot(str)
    def setNistReferencePath(self, qt_file_path):
        url = QUrl(qt_file_path)
        file_path = ""
        if url.isValid():
            file_path = url.toLocalFile()
        else:
            self.errorOccurred.emit("Error, File Path Not Valid")
            return

        # Ensure correct extension
        if not file_path.endswith(".csv"):
            file_path += ".csv"

        # Assign the path to the Helium Neon Line Data to the input file path.
        # Then check if all other file paths have been inputted and the central wavelength set. If yes, let UI know that user can initiate calibration
        self.Nist_Reference_Path = file_path
        if(self.isCalibrationReady()):
            self.canCalibrate.emit(True)

    @Slot(str)
    def setHeNePath(self, qt_file_path):
        url = QUrl(qt_file_path)
        file_path = ""
        if url.isValid():
            file_path = url.toLocalFile()
        else:
            self.errorOccurred.emit("Error, File Path Not Valid")
            return

        # Ensure correct extension
        if not file_path.endswith(".npy"):
            file_path += ".npy"

        # Assign the path to the Helium Neon Line Data to the input file path.
        # Then check if all other file paths have been inputted and the central wavelength set. If yes, let UI know that user can initiate calibration
        self.He_Ne_Path = file_path
        if(self.isCalibrationReady()):
            self.canCalibrate.emit(True)

    @Slot(str)
    def setNe633AnchorPath(self, qt_file_path):
        url = QUrl(qt_file_path)
        file_path = ""
        if url.isValid():
            file_path = url.toLocalFile()
        else:
            self.errorOccurred.emit("Error, File Path Not Valid")
            return

        # Ensure correct extension
        if not file_path.endswith(".npy"):
            file_path += ".npy"

        # Assign the path to the 633 nm Neon Anchor Data to the input file path.
        # Then check if all other file paths have been inputted and the central wavelength set. If yes, let UI know that user can initiate calibration
        self.Ne_633_Anchor_Path = file_path
        if(self.isCalibrationReady()):
            self.canCalibrate.emit(True)

    @Slot(str)
    def setNeCalibrationPath(self, qt_file_path):
        url = QUrl(qt_file_path)
        file_path = ""
        if url.isValid():
            file_path = url.toLocalFile()
        else:
            self.errorOccurred.emit("Error, File Path Not Valid")
            return

        # Ensure correct extension
        if not file_path.endswith(".npy"):
            file_path += ".npy"

        # Assign the path to the Neon Data we need to calibrate to the input file path.
        # Then check if all other file paths have been inputted and the central wavelength set. If yes, let UI know that user can initiate calibration
        self.Ne_Calibration_Path = file_path
        if(self.isCalibrationReady()):
            self.canCalibrate.emit(True)

    @Slot(int)
    def setCentralWavelength(self, approximate_wavelength):
        self.approximate_central_wavelength = approximate_wavelength

    @Slot()
    def calibrateCamera(self):
        if self.is_live_capture_running or self.is_standard_capture_running:
            self.errorOccurred.emit("Cannot calibrate while acquiring data")
        else:
            self.wavelength_calibrator.Reset()
            self.wavelength_calibrator.Calibrate(self.Ne_633_Anchor_Path, self.He_Ne_Path, self.Nist_Reference_Path)
            self.calibrationStatus, self.max_residual = self.wavelength_calibrator.Central_Wavelength_Shift(self.Ne_Calibration_Path, self.approximate_central_wavelength)
            self.min_wavelength, self.max_wavelength = self.wavelength_calibrator.Get_Wavelength_Range()

            if self.calibrationStatus:
                self.isCalibrated.emit("Calibrated")
            else:
                self.isCalibrated.emit("Uncalibrated")

            self.centralWavelengthChanged.emit(self.wavelength_calibrator.Get_Central_Wavelength())
            self.wavelengthRangeChanged.emit(self.min_wavelength, self.max_wavelength)
            self.residualCalculated.emit(self.max_residual)

            self.isCalibrationReadyToUse = True
            self.calibration_save_enabled = True

            free_parameters = self.wavelength_calibrator.Get_Free_Parameters()
            self.calibrationVariablesRequested.emit(free_parameters['Pc'], free_parameters['Dv'], free_parameters['gamma'], free_parameters['lambda_c'])
            self.maxResidualRequested.emit(self.max_residual)
            self.canSaveCalibrationChanged.emit(True)

    @Slot(str)
    def loadCalibrationFile(self, qt_file_path):
        if self.is_live_capture_running or self.is_standard_capture_running:
            self.errorOccurred.emit("Cannot load calibration file while acquiring data")
        else:
            url = QUrl(qt_file_path)
            file_path = ""
            if url.isValid():
                file_path = url.toLocalFile()
            else:
                self.errorOccurred.emit("Error, File Path Not Valid")
                return

            # Ensure correct extension
            if not (file_path.endswith(".csv") or file_path.endswith(".hdf5") or file_path.endswith(".h5")):
                self.errorOccurred.emit("Unsupported background file type")
                return

            Pc = 0
            Dv = 0
            gamma = 0
            lambda_c = 0
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
                Pc = df['Pc'][0]
                Dv = df['Dv'][0]
                gamma = df['gamma'][0]
                lambda_c = df['lambda_c'][0]

            elif file_path.endswith(".hdf5") or file_path.endswith(".h5"):
                with h5py.File(file_path, "r") as f:
                    if "wavelength_calibration_applied" not in list(f.attrs.keys()):
                        raise KeyError("HDF5 file does not contain wavelength calibration attributes")

                    Pc = f.attrs["central_pixel"]
                    Dv = f.attrs["deviation_angle (rads)"]
                    gamma = f.attrs["tilt_angle (rads)"]
                    lambda_c = f.attrs["central_wavelength (nm)"]

            self.wavelength_calibrator.Set_Free_Parameters({'Pc': Pc, 'Dv': Dv, 'gamma': gamma, 'lambda_c': lambda_c})
            self.calibrationVariablesRequested.emit(Pc, Dv, gamma, lambda_c)
            self.isCalibrationReadyToUse = True

            self.min_wavelength, self.max_wavelength = self.wavelength_calibrator.Get_Wavelength_Range()

            self.centralWavelengthChanged.emit(self.wavelength_calibrator.Get_Central_Wavelength())
            self.wavelengthRangeChanged.emit(self.min_wavelength, self.max_wavelength)
            self.notApplicableResidual.emit()
            self.isCalibrated.emit("Loaded")

    @Slot(str)
    def saveCalibrationFile(self, qt_save_path):
        url = QUrl(qt_save_path)
        save_path = ""
        if url.isValid():
            save_path = url.toLocalFile()
        else:
            self.errorOccurred.emit("Error, File Path Not Valid")
            return

        if self.calibration_save_enabled:
            if not save_path.endswith(".csv"):
                save_path += ".csv"

            optimal_parameters = self.wavelength_calibrator.Get_Free_Parameters()
            optimal_parameters['Dv (deg)'] = optimal_parameters['Dv'] * 180/np.pi
            optimal_parameters['gamma (deg)'] = optimal_parameters['gamma'] * 180/np.pi

            df = pd.DataFrame(optimal_parameters, index = [0])
            df.to_csv(save_path)

    @Slot()
    def request_capture_background(self):
        if self.is_live_capture_running or self.is_standard_capture_running:
            self.errorOccurred.emit("Cannot capture new background while acquiring data")
        else:
            self.is_capturing_background = True

            self.captureBackgroundRequested.emit(self.background_n_frames)
            self.enableBackgroundSubtractionRequested.emit(False)

            self.canSaveBackgroundChanged.emit(False)
            self.canSubtractBackgroundChanged.emit(False)
            self.canResetBackgroundChanged.emit(False)

    @Slot()
    def capture_background_complete(self):
        self.canSaveBackgroundChanged.emit(True)
        self.canSubtractBackgroundChanged.emit(True)
        self.is_capturing_background = False

    @Slot()
    def request_subtract_background(self):
        """Enable background subtraction"""
        if self.is_live_capture_running or self.is_standard_capture_running:
            self.errorOccurred.emit("Cannot subtract new background while acquiring data")
        else:
            self.enableBackgroundSubtractionRequested.emit(True)

            self.canSubtractBackgroundChanged.emit(False)
            self.canResetBackgroundChanged.emit(True)

    @Slot()
    def reset_background(self):
        if self.is_live_capture_running or self.is_standard_capture_running:
            self.errorOccurred.emit("Cannot reset background subtraction while acquiring data")
        else:
            """Enable background subtraction"""
            self.enableBackgroundSubtractionRequested.emit(False)

            self.canSubtractBackgroundChanged.emit(True)
            self.canResetBackgroundChanged.emit(False)

    @Slot(str)
    def request_save_background(self, qt_save_path):
        url = QUrl(qt_save_path)
        save_path = ""
        if url.isValid():
            save_path = url.toLocalFile()
        else:
            self.errorOccurred.emit("Error, File Path Not Valid")
            return

        if not save_path.endswith(".npy"):
            save_path += ".npy"
        self.saveBackgroundRequested.emit(save_path)

    @Slot(str)
    def load_background_requested(self, qt_file_path):
        if self.is_live_capture_running or self.is_standard_capture_running:
            self.errorOccurred.emit("Cannot load new background while acquiring data")
        else:
            url = QUrl(qt_file_path)
            file_path = ""
            if url.isValid():
                file_path = url.toLocalFile()
            else:
                self.errorOccurred.emit("Error, File Path Not Valid")
                return

            # Ensure correct extension
            if not (file_path.endswith(".npy") or file_path.endswith(".hdf5") or file_path.endswith(".h5")):
                self.errorOccurred.emit("Unsupported background file type")
                return

            self.loadBackgroundRequested.emit(file_path)
            self.enableBackgroundSubtractionRequested.emit(False)

            self.canSubtractBackgroundChanged.emit(True)

    @Slot()
    def request_snapshot(self):
        self.snapshotRequested.emit()

    @Slot(str)
    def request_save_snapshot(self, qt_file_path):
        url = QUrl(qt_file_path)
        file_path = ""
        if url.isValid():
            file_path = url.toLocalFile()
        else:
            self.errorOccurred.emit("Error, File Path Not Valid")
            return

        # Ensure correct extension
        if not file_path.endswith(".npy"):
            file_path += ".npy"

        self.saveSnapshotRequested.emit(file_path)

    @Slot(bool)
    def request_can_save_raw(self, status):
        self.save_raw_enabled = status
        self.canSaveRawRequested.emit(status)

    @Slot(bool)
    def request_can_save_wavelength(self, status):
        if self.isCalibrationReadyToUse:
            self.save_wavelength_enabled = status
            self.canSaveWavelengthRequested.emit(status)
        else:
            self.canEnableWavelengthCheckBox.emit(False)
            self.errorOccurred.emit("Must calibrate or load calibration file to save wavelength data")

    @Slot(str)
    def setStandardCaptureSavePath(self, qt_file_path):
        url = QUrl(qt_file_path)
        file_path = ""
        if url.isValid():
            file_path = url.toLocalFile()
        else:
            self.errorOccurred.emit("Error, File Path Not Valid")
            return

        # Ensure correct extension
        if not file_path.endswith(".hdf5"):
            file_path += ".hdf5"

        self.standardCaptureSavePathRequested.emit(file_path)

    @Slot(str)
    def setLiveCaptureSavePath(self, qt_file_path):
        url = QUrl(qt_file_path)
        file_path = ""
        if url.isValid():
            file_path = url.toLocalFile()
        else:
            self.errorOccurred.emit("Error, File Path Not Valid")
            return

        # Ensure correct extension
        if not file_path.endswith(".hdf5"):
            file_path += ".hdf5"

        self.liveCaptureSavePathRequested.emit(file_path)

    @Slot()
    def request_standard_capture(self):
        if self.is_live_capture_running:
            self.errorOccurred.emit("Cannot initialize standard capture when live capture is running")
        elif self.is_capturing_background:
            self.errorOccurred.emit("Cannot initialize standard capture while capturing background")
        elif not self.save_raw_enabled and not self.save_wavelength_enabled:
            self.errorOccurred.emit("No data to collect chosen")
        else:
            self.is_standard_capture_running = True
            self.standardCaptureRequested.emit(self.standard_capture_n_frames)
            self.canEditControls.emit(False)
            self.canStandardCaptureChanged.emit(False)

    @Slot()
    def end_standard_capture(self):
        self.canEditControls.emit(True)
        self.is_standard_capture_running = False

    @Slot()
    def request_live_capture_start(self):
        if self.is_standard_capture_running:
            self.errorOccurred.emit("Cannot initialize live capture when standard capture is unfinished")
        elif self.is_capturing_background:
            self.errorOccurred.emit("Cannot initialize live capture while capturing background")
        elif not self.save_raw_enabled and not self.save_wavelength_enabled:
            self.errorOccurred.emit("No data to collect chosen")
        else:
            self.is_live_capture_running = True
            self.startLiveCaptureRequested.emit()
            self.canEditControls.emit(False)
            self.canEditLiveCaptureFileChanged.emit(False)
            self.canStartLiveCaptureChanged.emit(False)
            self.canEndLiveCaptureChanged.emit(True)

    @Slot()
    def request_live_capture_end(self):
        self.is_live_capture_running = False
        self.endLiveCaptureRequested.emit()
        self.canEditControls.emit(True)

    # Capturing frames and saving them to storage is computationally expensive
    # We therefore offload all frame captures and file I/O to a worker thread
    def start_camera_worker(self):
        # Create a thread and attach it to a CameraWorker object
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker(self.cam, self.max_width, self.max_height)
        self.camera_worker.moveToThread(self.camera_thread)

        # When started, call the Camera worker start function
        # Connect the emitted frames from the CameraWorker to the emitted frames from the CameraController
        self.camera_thread.started.connect(self.camera_worker.start)
        self.camera_worker.frameReady.connect(self.frameReady.emit)

        # Connect to all the functions that the CameraController needs to access within the Camera Worker
        self.gainRequested.connect(self.camera_worker.get_gain)
        self.exposureRequested.connect(self.camera_worker.get_exposure)
        self.targetTempRequested.connect(self.camera_worker.get_target_temp)
        self.currentTempRequested.connect(self.camera_worker.get_current_temp)
        self.maxResidualRequested.connect(self.camera_worker.get_max_residual)
        self.coolerStatusRequested.connect(self.camera_worker.get_cooler_status)
        self.antiDewHeaterStatusRequested.connect(self.camera_worker.get_antiDewHeater_status)
        self.captureBackgroundRequested.connect(self.camera_worker.capture_background)
        self.enableBackgroundSubtractionRequested.connect(self.camera_worker.enable_subtraction)
        self.saveBackgroundRequested.connect(self.camera_worker.save_background)
        self.loadBackgroundRequested.connect(self.camera_worker.load_background)
        self.calibrationVariablesRequested.connect(self.camera_worker.set_wavelength_calibration_variables)
        self.canSaveRawRequested.connect(self.camera_worker.set_can_save_raw)
        self.canSaveWavelengthRequested.connect(self.camera_worker.set_can_save_wavelength)
        self.standardCaptureSavePathRequested.connect(self.camera_worker.set_standard_capture_save_path)
        self.liveCaptureSavePathRequested.connect(self.camera_worker.set_live_capture_save_path)
        self.startLiveCaptureRequested.connect(self.camera_worker.start_live_capture)
        self.endLiveCaptureRequested.connect(self.camera_worker.end_live_capture)
        self.standardCaptureRequested.connect(self.camera_worker.start_standard_capture)
        self.snapshotRequested.connect(self.camera_worker.capture_snapshot)
        self.saveSnapshotRequested.connect(self.camera_worker.save_snapshot)

        # Connect all functions that the Camera Worker needs access to within the Camera Controller
        self.camera_worker.standardCaptureFinished.connect(self.end_standard_capture)
        self.camera_worker.captureBackgroundFinished.connect(self.capture_background_complete)

        # When finished, stop the thread, delete the camera worker, and delete the thread
        self.camera_worker.finished.connect(self.camera_thread.quit)
        self.camera_worker.finished.connect(self.camera_worker.deleteLater)
        self.camera_thread.finished.connect(self.camera_thread.deleteLater)

        # Officially start the thread
        self.camera_thread.start()

    def stop_worker(self):
        self._worker.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()

    provider = CameraImageProvider()
    engine.addImageProvider("camera", provider)

    cam = CameraController()
    cam.start_camera_worker()
    cam.frameReady.connect(provider.updateImage)
    engine.rootContext().setContextProperty("cameraController", cam)

    engine.load("main.qml")
    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec())
