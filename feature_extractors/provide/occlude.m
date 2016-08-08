function occludedImages = occlude(images, ...
    numsBubbles, bubbleCenters, bubbleSigmas)

occludedImages = cell(length(images), 1);
for i = 1:length(images)
    numBubbles = numsBubbles(i);
    S.c = bubbleCenters(i, 1:numBubbles);
    S.sig = bubbleSigmas(i, 1:numBubbles);
    occludedImages{i} = AddBubble(images{i}, S);
end
