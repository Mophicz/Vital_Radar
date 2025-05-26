%% Converts h5 file from python to matlab struct
clear
clc
close all

filename = 'data/signals.h5';

% Read file structure to detect all frame groups
fileInfo = h5info(filename);
frames = fileInfo.Groups;
N_st = length(frames);

% 13, 8, 11, 8

% Prepare structure to store data
allData = struct();
N_pairs = 40;

for p = 1:N_pairs
    signals = zeros(8192, N_st);
    sampling_rate = zeros(N_st, 1);
    for f = 1:N_st
        groupPath = frames(f).Groups(p).Name;

        time_signal = h5read(filename, [groupPath '/amplitudes']);
        amplitudes = h5read(filename, [groupPath '/timeAxis']);

        dt = mean(diff(time_signal));
        fs_ft = 1 / dt;

        signals(:, f) = amplitudes;
        sampling_rate(f) = fs_ft;
    end
    allData(p).data = signals;
    allData(p).fs_ft = mean(sampling_rate);
end

save('data/radar_data.mat', 'allData');


%% Example: downsampling fast time signal with plots
close all;
clear;
clc;

load("data\radar_data.mat");

Fs = 102.4e9;              
Fc = 7.15e9;                        
B = 1.7e9;   

% Extract raw signal and parameters from struct
x = allData(1).data(:,1);                      

N = length(x);
n = 1:N;

% Downconvert to baseband
x_bb = x .* exp(-1j * 2 * pi * Fc * n' / Fs);
% PSD raw-signal
Pxx = abs(fft(x)).^2 / (N*Fs);     % PSD
% fft baseband-signal
Xbb = fft(x_bb);
% PSD baseband-signal
Pbb = abs(Xbb).^2 / (N*Fs);  % baseband PSD

Pxx = fftshift(Pxx);
Pbb = fftshift(Pbb);

% Downsampling baseband-signal
Nds = N * B/Fs;
Xbb = fftshift(Xbb);
Xbb_downsampled = Xbb(N/2+1-Nds/2:N/2+1+Nds/2);
Pbb_downsampled = abs(Xbb_downsampled).^2 / (N*Fs);
% ifft downsampled signal
x_bb_downsampled = ifft(Xbb_downsampled) .* Nds/N;

% frequency vector
f = -(Fs/2):Fs/(N-1):(Fs/2);
f = f .* 1/1e9;

% Plotting
figure;
t = tiledlayout(2,3);

nexttile
plot(x)
xlim([1 8200])
title('Raw (tx1, rx1)')
xlabel('Fast Time Sample n')
ylabel('s_l(n)')

nexttile
plot(real(x_bb))
hold on
plot(imag(x_bb))
xlim([1 8200])
title('Baseband (tx1, rx1)')
xlabel('Fast Time Sample n')
ylabel('s_{l}(n)')
legend('Real', 'Imaginary')

nexttile
plot(real(x_bb_downsampled))
hold on
plot(imag(x_bb_downsampled))
xlim([1 137])
title('Downsampled (tx1, rx1)')
xlabel('Fast Time Sample n')
ylabel('s_{l}(n)')
legend('Real', 'Imaginary')

nexttile
plot(f, 20*log(Pxx))
hold on
xlim([0 10])
xlabel('Frequency [GHz]')
ylabel('20 log(P_{ss}(e^{j\omega}))')

nexttile
plot(f, 20*log(Pbb))
xlim([-5 5])
xlabel('Frequency [GHz]')
ylabel('20 log(P_{ss}(e^{j\omega}))')

nexttile
plot(f(N/2+1-Nds/2:N/2+1+Nds/2), 20*log(Pbb_downsampled))
xlim([-5 5])
xlabel('Frequency [GHz]')
ylabel('20 log(P_{ss}(e^{j\omega}))')

% Adjust the layout to minimize extra space
t.TileSpacing = 'compact';
t.Padding = 'compact';

% Define figure size
set(gcf, 'Units', 'inches', 'Position', [1 1 12 8]);

% Export the figure with tight layout
exportgraphics(gcf, 'FT_downsampling.png', 'Resolution', 300);

%% Downsampling all fast time signals for one antenna pair
close all;
clear;
clc;

Fs = 102.4e9;              
Fc = 7.15e9;                        
B = 1.7e9;

load("data\radar_data.mat");

x = allData(1).data; % 8192x89 double (columns: fast time, rows: slow time)
[N, L] = size(x);
Nds = N * B/Fs;

y = zeros(Nds+1, L);
for l = 1:L
    y(:, l) = downsample(x(:,l), Fs, Fc, B);
end

k1 = 1;
k2 = 3;

[U, S, V] = svd(y);

U_DC = U(:, k1);
S_DC = S(1:k1, 1:k1);
V_DC = V(:, 1:k1).';
X_DC = U_DC * S_DC * V_DC;

U_VS = U(:, k1+1:k2);
S_VS = S(k1+1:k2, k1+1:k2);
V_VS = V(:, k1+1:k2).';
X_VS = U_VS * S_VS * V_VS;

U_N = U(:, k2+1:end);
S_N = S(k2+1:end, k2+1:end);
V_N = V(:, k2+1:end).';
X_N = U_N * S_N * V_N;

% Plotting
figure;
t = tiledlayout(1, 3);
nexttile
imagesc(abs(X_DC))
nexttile
imagesc(abs(X_VS))
nexttile
imagesc(abs(X_N))