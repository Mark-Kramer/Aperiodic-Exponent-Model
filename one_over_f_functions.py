# Load necessary packages
import numpy as np
import matplotlib.pyplot as plt

# Compute the spectrum.
def compute_spectrum(x,t):                         # Compute the spectrum of signal x, time axis t in [s].
    N   = np.size(x)                               # Number of data points
    dt  = t[2]-t[1]                                # Time resolution, in [s].
    T   = t[-1]                                    # Total time of data, in [s].
    xf  = np.fft.fft(np.hanning(N)*(x-np.mean(x))) # Fourier transform of data, Hanning taper, 0-mean.
    S   = np.real(2*dt**2/T*(xf*np.conj(xf)))      # Spectrum
    S   = S[1:int(N/2)+1]                          # Keep only non-negative frequencies
    df  = 1/T                                      # Frequency resolution, in [Hz]
    fNQ = 1/dt/2                                   # Nyquist frequency, in [Hz]
    f   = np.arange(0,fNQ,df)                      # Frequency axis, in [Hz]
    return S, f

# Compute the aperiodic exponent.
def estimate_aperiodic_exponent(S,f,finterval):
    freq_interval_to_fit = (f >= finterval[0]) & (f<=finterval[1])      # For this frequency range,
                                                                        # Fit linear model: log10(S) vs log10(f)
    linear_fit           = np.polyfit(np.log10(f[freq_interval_to_fit]), np.log10(S[freq_interval_to_fit]), 1)
    x_linear_fit         = np.log10(f[freq_interval_to_fit])            # Return x-axis of fit.
    y_linear_fit         = linear_fit[1] + linear_fit[0]*x_linear_fit   # Return y-axis of fit.
    aperiodic_exponent   = linear_fit[0]                                # Return aperiodic exponent.
    return aperiodic_exponent, x_linear_fit, y_linear_fit

# Plot trace and spectrum with fit.
def make_plots(t,x, f,S, x_linear_fit,y_linear_fit,aperiodic_exponent, axX,axS):
    axX.plot(t,x, 'k');                              axX.set(xlabel="Time [s]", ylabel='[a.u.]')
    axS.plot(np.log10(f[2:]), np.log10(S[2:]), 'k'); axS.set(xlabel="Log$_{10}$(Frequency [Hz]", ylabel="Log$_{10}$(P)")
    axS.plot(x_linear_fit, y_linear_fit, 'r');       axS.text(x_linear_fit[0],y_linear_fit[0]+2, "Aperiodic exponent: %.2f" % aperiodic_exponent);
    axX.spines["top"].set_visible(False); axX.spines["right"].set_visible(False)
    axS.spines["top"].set_visible(False); axS.spines["right"].set_visible(False)
    
# Plot aperiodic exponent versus noise.
def make_plot_ae_vs_noise(res):
    aperiodic_exponents = res["aperiodic_exponents"]
    noise_values      = res["noise_values"]
    
    bounds = np.quantile(aperiodic_exponents,[0.025,0.975],1)
    mean   = np.mean(aperiodic_exponents,1)

    plt.plot(np.squeeze(noise_values),mean,'k')
    plt.plot(np.squeeze(noise_values),bounds[0,:],'red')
    plt.plot(np.squeeze(noise_values),bounds[1,:],'red')
    plt.fill_between(np.squeeze(noise_values),bounds[0,:],bounds[1,:], facecolor='red', alpha=0.5)
    plt.grid(True);
    plt.xlabel('Noise'); plt.ylabel('Aperiodic Exponent');