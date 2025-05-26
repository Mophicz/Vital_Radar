function [y] = downsample(x, Fs, Fc, B)
%DOWNSAMPLE Converts a fast time signal to baseband and downsamples to the
%radars active frequency band.
%   First the signal is converted to baseband by multiplying it with a 
%   complex exponential at the carrier frequency. Then a DFT is applied. 
%   The number of samples in the radar-bandwidth is calculated and used to
%   truncate the signal to only include the radar bandwidth. This remaining
%   signal is tranformed back using iDFT and returned.

N = length(x);
n = 1:N;

% Downconvert to baseband
xbb = x .* exp(-1j * 2 * pi * Fc * n' / Fs);

% DFT
Xbb = fft(xbb);

% Downsample (by truncating)
Nds = N * B/Fs; % number of samples in radar-bandwidth
Xbb = fftshift(Xbb); % shift 0 to middle of vector
Y = Xbb(N/2+1-Nds/2:N/2+1+Nds/2); % truncate outside of bandwidth region
Y = ifftshift(Y);

% iDFT
y = ifft(Y) .* Nds/N;
end