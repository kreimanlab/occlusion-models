function spectra = meanSpectra(images)
spectra = NaN([numel(images), size(images{1})]);
for i = 1:numel(images)
    spectra(i, :, :) = abs(fft2(mat2gray(images{i})));
end
spectra = squeeze(mean(spectra));
end