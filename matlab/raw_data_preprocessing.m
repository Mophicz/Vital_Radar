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

%%
close all;
clear;
clc;

load("..\data\radar_data.mat");

Fs = 102.4e9;              
Fc = 7.15e9;                        
B = 1.7e9;   

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

% Extract raw signal and parameters from struct
x = allData(1).data(:,1);   
% Helper function to export a figure tightly as vector PDF
function export_tight_vector(fig, filename)
    tightfig(fig);  % Requires tightfig.m in MATLAB path
    exportgraphics(fig, filename, ...
        'ContentType', 'vector', ...
        'BackgroundColor', 'none');
end

% === Plot 1: Raw signal ===
fig1 = figure;
plot(x)
xlim([1 8200])
xlabel('fast-time-sample n', 'FontSize', 18)
ylabel('s_l(n)', 'FontSize', 18)
set(fig1, 'Units', 'inches', 'Position', [1 1 6 4]);
export_tight_vector(fig1, 'raw_signal.pdf');

% === Plot 2: Baseband signal ===
fig2 = figure;
plot(real(x_bb))
hold on
plot(imag(x_bb))
xlim([1 8200])
xlabel('fast-time-sample n', 'FontSize', 18)
ylabel('s_{l}(n)', 'FontSize', 18)
legend('Reell', 'Imaginär')
set(fig2, 'Units', 'inches', 'Position', [1 1 6 4]);
export_tight_vector(fig2, 'baseband_signal.pdf');

% === Plot 3: Downsampled baseband ===
fig3 = figure;
plot(real(x_bb_downsampled))
hold on
plot(imag(x_bb_downsampled))
xlim([1 137])
xlabel('fast-time-sample n', 'FontSize', 18)
ylabel('s_{l}(n)', 'FontSize', 18)
legend('Reell', 'Imaginär')
set(fig3, 'Units', 'inches', 'Position', [1 1 6 4]);
export_tight_vector(fig3, 'downsampled_signal.pdf');

% === Plot 4: PSD of raw signal ===
fig4 = figure;
plot(f, 20*log10(Pxx))
xlim([0 10])
xlabel('Frequenz [GHz]', 'FontSize', 18)
ylabel('20 log(P_{ss}(e^{j\omega}))', 'FontSize', 18)
set(fig4, 'Units', 'inches', 'Position', [1 1 6 4]);
export_tight_vector(fig4, 'psd_raw.pdf');

% === Plot 5: PSD of baseband signal ===
fig5 = figure;
plot(f, 20*log10(Pbb))
xlim([-5 5])
xlabel('Frequenz [GHz]', 'FontSize', 18)
ylabel('20 log(P_{ss}(e^{j\omega}))', 'FontSize', 18)
set(fig5, 'Units', 'inches', 'Position', [1 1 6 4]);
export_tight_vector(fig5, 'psd_baseband.pdf');

% === Plot 6: PSD of downsampled baseband ===
fig6 = figure;
plot(f(N/2+1-Nds/2:N/2+1+Nds/2), 20*log10(Pbb_downsampled))
xlim([-5 5])
xlabel('Frequenz [GHz]', 'FontSize', 18)
ylabel('20 log(P_{ss}(e^{j\omega}))', 'FontSize', 18)
set(fig6, 'Units', 'inches', 'Position', [1 1 6 4]);
export_tight_vector(fig6, 'psd_downsampled.pdf');


%% SVD decluttering
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

%%
close all;
clear;
clc;

Fs = 102.4e9;              
Fc = 7.15e9;                        
B = 1.7e9;
K = 137;
T = 10;

load("C:\Users\Michael\Projects\Projektseminar_Medizintechnik\radar_dataset_DataPort\m0_radar.mat");

x = data_radar.tx_1(:, :, 1);
[N, L] = size(x);
Nds = N * B/Fs;
Ff = L / T;

c = physconst('LightSpeed');
dF = B / K;
n = 4; 
N = 137;

%d = (n * c) / (2 * N * dF)

x = x - movmean(x, 50, 2);

v = var(x(:, 1:45), 0, 2);

plot(v)