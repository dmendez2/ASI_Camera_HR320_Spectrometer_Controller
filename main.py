import os
import sys
import cv2
import h5py
import numpy as np
import pandas as pd
import zwoasi as asi

from lmfit import minimize, Parameters

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from PySide6.QtQuick import QQuickImageProvider
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject, Signal, Slot, QTimer, QUrl

class mockCamera():
    def __init__(self):
        self.camera_properties = {"MaxHeight": 1200, "MaxWidth": 1200}
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

    def __init__(self, Nist_Reference_Path):
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

        ### The Reference Ne I spectra from NIST (And Preprocessing of the Spectra) ###
        self.reference_spectra = pd.read_csv(Nist_Reference_Path)[['obs_wl_air(nm)', 'intens', 'Aki(s^-1)']].astype(float).dropna() # Reference Ne I spectral line wavelength --> From NIST (mm)
        self.reference_spectra['obs_wl_air(nm)'] *= 1e-6 # Convert wavelengths to millimeters for later calculations (mm)
        self.reference_spectra.rename(columns = {'obs_wl_air(nm)': 'obs_wl_air(mm)'}, inplace = True) # Rename column to reflect unit change

        ### Other Relavent Variables Related to the NIST Reference Spectra ###
        self.transition_probability_threshold = 4.0 * 1e6 # Apparent minimum transition rate probability detectable by spectrometer (1/s)
        self.intensity_threshold = 5000 # Apparent minimum relative intensity detectable by spectrometer

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

    def Extract_Peak_Centers(self, data, prominence = 2):
        # Collapse 2D CCD data to 1D spectra by taking median of each column
        spectrum = np.mean(data, axis = 0)

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
        intensity = intensity/np.trapz(intensity)
        return intensity

    # Utilizing Equations from "The Optics of Spectroscopy" by J.M. Lerner and A. Thevenon to Compute the Wavelength For a Given Pixel Detection
    def Get_Calibrated_Wavelength_From_Pixel(self, data):
        # Compute the average, integration normalized intensity along the X-Axis
        intensity = self.Process_Pixel_Intensity(data)

        # Create an array for every single X-axis pixel
        pixels = np.arange(self.P_Min, self.P_Max + 1, 1)

        # Return the wavelengths (nm) multiplied by the intensity
        return self.Get_Wavelength_From_Pixel(self.k, self.n, self.F, self.Dv, self.gamma, self.Pw, self.Pc, self.lambda_c, pixels) * intensity * 1e6

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
    def Calibrate(self, Ne_Data_Path, He_Ne_Line_Path):
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
        spectrum, peaks = self.Extract_Peak_Centers(He_Ne_Line, prominence = 10)
        self.Pc = peaks[0]

        # Extract the Peak Centers from the measured Ne Spectrum
        spectrum, P_measured = self.Extract_Peak_Centers(Ne_Data, prominence = 2)

        # Compute the minimum wavelength that can be resolved from our diffraction grating and ensure the residuals are smaller than it. Set initial residuals to infinity
        # If they are, calibration is complete. Otherwise, if 5 rounds of calibration pass without success, we consider the calibration a failure.
        d_lambda = self.Dlambda(self.n, self.width, self.lambda_c)
        residuals = np.ones(len(P_measured)) * np.inf
        iter = 0
        while(np.all(residuals > d_lambda) and iter < 5):
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

        if(np.all(residuals < d_lambda)):
            print("633 Calibration Complete")
        else:
            print("633 Calibration Failed")
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
        spectrum, P_measured = self.Extract_Peak_Centers(data, prominence = 2.0)

        # Compute the minimum wavelength that can be resolved from our diffraction grating and ensure the residuals are smaller than it. Set initial residuals to infinity
        # If they are, calibration is complete. Otherwise, if 5 rounds of calibration pass without success, we consider the calibration a failure.
        d_lambda = self.Dlambda(self.n, self.width, self.lambda_c)
        residuals = np.ones(len(P_measured)) * np.inf
        iter = 0
        while(np.all(residuals > d_lambda) and iter < 5):
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
            print(lambda_c, " Calibration Complete")
            return True, np.max(residuals)*1e6
        else:
            print(lambda_c, " Calibration Failed")
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

class CameraController(QObject):
    frameReady = Signal(QImage)
    canSaveBackground = Signal(bool)
    gainRangeChanged = Signal(int, int)
    exposureRangeChanged = Signal(int, int)
    tempRangeChanged = Signal(int, int)
    temperatureChanged = Signal(float)
    residualCalculated = Signal(float)
    centralWavelengthChanged = Signal(float)
    wavelengthRangeChanged = Signal(float, float)
    canSubtractBackgroundChanged = Signal(bool)
    canResetBackgroundChanged = Signal(bool)
    canSaveCalibrationChanged = Signal(bool)
    canCalibrate = Signal(bool)
    isCalibrated = Signal(bool)
    liveCaptureStarted = Signal()
    liveCaptureStopped = Signal()

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
        print(info)
        self.max_height = info['MaxHeight']
        self.max_width = info['MaxWidth']
        print(self.max_height)
        print(self.max_width)
        self.cam.set_roi_format(self.max_width, self.max_height, 1, asi.ASI_IMG_RAW16)

        controls = self.cam.get_controls()
        self.exposure_min = controls['Exposure']['MinValue']//1000
        self.exposure_max = min(controls['Exposure']['MaxValue']//1000, 5000)
        self.gain_min = max(controls['Gain']['MinValue'], 0)
        self.gain_max = controls['Gain']['MaxValue']
        self.temp_min = controls['TargetTemp']['MinValue']
        self.temp_max = controls['TargetTemp']['MaxValue']
        self.n_frames = 1

        self.control_types = dict()
        for control in controls.keys():
            self.control_types[control] = controls[control]['ControlType']
        print(controls.keys())
        print(self.control_types)

        self.cam.start_video_capture()

        # Background
        self.background = None
        self.subtraction_enabled = False  # only true after pressing subtract

        # Calibration
        self.wavelength_calibrator = None
        self.calibrationStatus = False
        self.calibration_save_enabled = False # Only true after a calibration has been initiated

        self.Nist_Reference_Path = None
        self.He_Ne_Path = None
        self.Ne_633_Anchor_Path = None
        self.Ne_Calibration_Path = None

        self.min_wavelength = 400
        self.approximate_central_wavelength = 700
        self.max_wavelength = 1000
        self.max_residual = 0

        # Live Capture
        self.data_save_path = None
        self.live_capture_thread = None
        self.h5file = None
        self.h5_dataset = None
        self.h5_frame_count = 0
        self.live_capture_running = False

        # Access cache to use previous calibrations
        if os.path.exists("cache/calibration.csv"):
            self.wavelength_calibrator.set_free_parameters(pd.read_csv("cache/calibration.csv"))

        # Timer for grabbing frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(500)

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
    def setNFrames(self, n):
        self.n_frames = n

    @Slot(int)
    def setExposure(self, ms):
        self.cam.set_control_value(self.control_types['Exposure'], max(1000, ms*1000))

    @Slot(int)
    def setGain(self, gain):
        self.cam.set_control_value(self.control_types['Gain'], gain)

    @Slot(int)
    def setTargetTemp(self, temp):
        self.cam.set_control_value(self.control_types['TargetTemp'], temp)

    @Slot()
    def update_temperature(self):
        try:
            temp_val = self.cam.get_control_value(self.control_types['Temperature'])[0]
            temp_c = temp_val // 10 # convert to °C
            self.temperatureChanged.emit(temp_c)
        except Exception as e:
            print("Error reading temperature:", e)

    @Slot(bool)
    def setCooler(self, enabled: bool):
        try:
            self.cam.set_control_value(self.control_types['CoolerOn'], 1 if enabled else 0)
            print("Cooler set to", enabled)
        except Exception as e:
            print("Error setting cooler:", e)

    @Slot(bool)
    def setAntiDewHeater(self, enabled: bool):
        try:
            self.cam.set_control_value(self.control_types['AntiDewHeater'], 1 if enabled else 0)
            print("Anti-dew heater set to", enabled)
        except Exception as e:
            print("Error setting anti-dew heater:", e)

    def isCalibrationReady(self):
        return (self.Nist_Reference_Path is not None) and (self.He_Ne_Path is not None) and (self.Ne_633_Anchor_Path is not None) and (self.Ne_Calibration_Path is not None)

    @Slot(str)
    def setNistReferencePath(self, qt_file_path):
        url = QUrl(qt_file_path)
        file_path = ""
        if url.isValid():
            file_path = url.toLocalFile()
        else:
            print("Error, File Path Not Valid")
            return

        # Ensure correct extension
        if not file_path.endswith(".csv"):
            file_path += ".csv"

        # Assign the path to the Helium Neon Line Data to the input file path.
        # Then check if all other file paths have been inputted and the central wavelength set. If yes, let UI know that user can initiate calibration
        self.Nist_Reference_Path = file_path
        if(self.isCalibrationReady):
            self.canCalibrate.emit(True)

    @Slot(str)
    def setHeNePath(self, qt_file_path):
        url = QUrl(qt_file_path)
        file_path = ""
        if url.isValid():
            file_path = url.toLocalFile()
        else:
            print("Error, File Path Not Valid")
            return

        # Ensure correct extension
        if not file_path.endswith(".npy"):
            file_path += ".npy"

        # Assign the path to the Helium Neon Line Data to the input file path.
        # Then check if all other file paths have been inputted and the central wavelength set. If yes, let UI know that user can initiate calibration
        self.He_Ne_Path = file_path
        if(self.isCalibrationReady):
            self.canCalibrate.emit(True)

    @Slot(str)
    def setNe633AnchorPath(self, qt_file_path):
        url = QUrl(qt_file_path)
        file_path = ""
        if url.isValid():
            file_path = url.toLocalFile()
        else:
            print("Error, File Path Not Valid")
            return

        # Ensure correct extension
        if not file_path.endswith(".npy"):
            file_path += ".npy"

        # Assign the path to the 633 nm Neon Anchor Data to the input file path.
        # Then check if all other file paths have been inputted and the central wavelength set. If yes, let UI know that user can initiate calibration
        self.Ne_633_Anchor_Path = file_path
        if(self.isCalibrationReady):
            self.canCalibrate.emit(True)

    @Slot(str)
    def setNeCalibrationPath(self, qt_file_path):
        url = QUrl(qt_file_path)
        file_path = ""
        if url.isValid():
            file_path = url.toLocalFile()
        else:
            print("Error, File Path Not Valid")
            return

        # Ensure correct extension
        if not file_path.endswith(".npy"):
            file_path += ".npy"

        # Assign the path to the Neon Data we need to calibrate to the input file path.
        # Then check if all other file paths have been inputted and the central wavelength set. If yes, let UI know that user can initiate calibration
        self.Ne_Calibration_Path = file_path
        if(self.isCalibrationReady):
            self.canCalibrate.emit(True)

    @Slot(int)
    def setCentralWavelength(self, approximate_wavelength):
        self.approximate_central_wavelength = approximate_wavelength

    @Slot()
    def calibrateCamera(self):
        self.wavelength_calibrator = WavelengthCalibrator(self.Nist_Reference_Path)
        self.wavelength_calibrator.Calibrate(self.Ne_633_Anchor_Path, self.He_Ne_Path)
        self.calibrationStatus, self.max_residual = self.wavelength_calibrator.Central_Wavelength_Shift(self.Ne_Calibration_Path, self.approximate_central_wavelength)
        self.min_wavelength, self.max_wavelength = self.wavelength_calibrator.Get_Wavelength_Range()

        self.isCalibrated.emit(self.calibrationStatus)
        self.centralWavelengthChanged.emit(self.wavelength_calibrator.Get_Central_Wavelength())
        self.wavelengthRangeChanged.emit(self.min_wavelength, self.max_wavelength)
        self.residualCalculated.emit(self.max_residual)

        self.calibration_save_enabled = True
        self.canSaveCalibrationChanged.emit(True)

    @Slot(str)
    def loadCalibrationFile(self, qt_file_path):
        url = QUrl(qt_file_path)
        file_path = ""
        if url.isValid():
            file_path = url.toLocalFile()
        else:
            print("Error, File Path Not Valid")
            return

        # Ensure correct extension
        if not file_path.endswith(".csv"):
            file_path += ".csv"
        df = pd.read_csv(file_path)
        self.wavelength_calibrator.set_free_parameters(df)
        print(f"Background opened from {file_path}")

    @Slot(str)
    def saveCalibrationFile(self, qt_save_path):
        url = QUrl(qt_save_path)
        save_path = ""
        if url.isValid():
            save_path = url.toLocalFile()
        else:
            print("Error, Save Path Not Valid")
            return

        if self.calibration_save_enabled:
            if not save_path.endswith(".csv"):
                save_path += ".csv"

            optimal_parameters = self.wavelength_calibrator.get_free_parameters()
            df = pd.DataFrame(optimal_parameters, index = [0])
            df.to_csv(save_path)

            # Save to cache as well
            cache_file_path = "cache/calibration.csv"
            df.to_csv(cache_file_path)
            print(f"Background saved to {save_path}")

    @Slot()
    def capture_background(self):
        """Capture the current n-averaged frame as background, optionally save to file."""
        frame = self.cam.get_video_data()
        frame = np.frombuffer(frame, dtype=np.uint16)
        frame = np.reshape(frame, (self.max_height, self.max_width))

        background = None
        for _ in range(self.n_frames):
            frame = self.cam.get_video_data()
            frame = np.frombuffer(frame, dtype=np.uint16)
            frame = np.reshape(frame, (self.max_height, self.max_width))
            background = frame if background is None else background + frame
        self.background = background / self.n_frames
        self.subtraction_enabled = False

        print("Background captured")
        self.canSaveBackground.emit(True)
        self.canSubtractBackgroundChanged.emit(True)

    @Slot()
    def subtract_background(self):
        """Enable background subtraction"""
        if self.background is not None:
            self.subtraction_enabled = True
            print("Background subtracted")
            self.canSubtractBackgroundChanged.emit(False)
            self.canResetBackgroundChanged.emit(True)

    @Slot()
    def reset_background(self):
        """Enable background subtraction"""
        self.subtraction_enabled = False
        self.canSubtractBackgroundChanged.emit(True)
        self.canResetBackgroundChanged.emit(False)

    @Slot(str)
    def save_background(self, qt_save_path):
        url = QUrl(qt_save_path)
        save_path = ""
        if url.isValid():
            save_path = url.toLocalFile()
        else:
            print("Error, Save Path Not Valid")
            return

        if self.background is not None:
            # Ensure correct extension
            if not save_path.endswith(".npy"):
                save_path += ".npy"
            np.save(save_path, self.background)
            print(f"Background saved to {save_path}")

    @Slot(str)
    def open_background(self, qt_file_path):
        url = QUrl(qt_file_path)
        file_path = ""
        if url.isValid():
            file_path = url.toLocalFile()
        else:
            print("Error, File Path Not Valid")
            return

        # Ensure correct extension
        if not file_path.endswith(".npy"):
            file_path += ".npy"
        self.background = np.load(file_path)
        self.subtraction_enabled = False
        self.canSubtractBackgroundChanged.emit(True)
        print(f"Background opened from {file_path}")

    @Slot(str)
    def setDataSavePath(self, qt_file_path):
        url = QUrl(qt_file_path)
        file_path = ""
        if url.isValid():
            file_path = url.toLocalFile()
        else:
            print("Error, File Path Not Valid")
            return

        # Ensure correct extension
        if not file_path.endswith(".hdf5"):
            file_path += ".hdf5"

        self.data_save_path = file_path

    @Slot()
    def start_live_capture(self):
        """
        Starts live capture and saves frames to HDF5 file at hdf5_path
        """
        if self.live_capture_running:
            print("Live capture already running")
            return

        # Open HDF5 file
        self.h5file = h5py.File(self.data_save_path, "w")
        self.h5_dataset = self.h5file.create_dataset(
            "frames",
            shape=(0, self.max_height, self.max_width),
            maxshape=(None, self.max_height, self.max_width),
            dtype=np.uint16,
            chunks=(1, self.max_height, self.max_width)  # chunk by frame
        )

        # Store metadata
        self.h5file.attrs["bit_depth"] = 16
        self.h5file.attrs["camera_width"] = self.max_width
        self.h5file.attrs["camera_height"] = self.max_height
        self.h5_frame_count = 0
        self.live_capture_running = True
        self.liveCaptureStarted.emit()
        print(f"Live capture started: saving to {self.data_save_path}")

    @Slot()
    def stop_live_capture(self):
        if not self.live_capture_running:
            return

        self.live_capture_running = False
        if self.h5file is not None:
            self.h5file.close()
            self.h5file = None
            self.h5_dataset = None
            print("Live capture stopped and HDF5 file closed")

        self.liveCaptureStopped.emit()

    def update_frame(self):
        frame = self.cam.get_video_data()
        frame = np.frombuffer(frame, dtype=np.uint16)
        frame = np.reshape(frame, (self.max_height, self.max_width))

        # Background subtraction
        if self.background is not None and self.subtraction_enabled:
            frame = frame - self.background
        frame[frame < 0] = 0

        # Q-Image expects an 8 bit display so we convert to from 16-bit to 8-bit
        # We do this by min-max normalizing and then multiplying by 255 (Max value 8-bits can express)
        frame_8_bit = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Convert to QImage for QML
        qimg = QImage(frame_8_bit.data, frame_8_bit.shape[1], frame_8_bit.shape[0], frame_8_bit.strides[0], QImage.Format_Grayscale8)
        self.frameReady.emit(qimg.copy())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()

    provider = CameraImageProvider()
    engine.addImageProvider("camera", provider)

    cam = CameraController()
    cam.frameReady.connect(provider.updateImage)
    engine.rootContext().setContextProperty("cameraController", cam)

    engine.load("main.qml")
    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec())
