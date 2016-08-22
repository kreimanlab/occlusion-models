function occludedImages = occludeImages(...
    numsBubbles, bubbleCenters, bubbleSigmas, images, dataSelection)
nums = numsBubbles(dataSelection);
centers = bubbleCenters(dataSelection, :);
sigmas = bubbleSigmas(dataSelection, :);
occludedImages = occlude(images, nums, centers, sigmas);
end
