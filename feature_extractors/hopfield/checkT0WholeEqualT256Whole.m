function checkT0WholeEqualT256Whole()
data = load('data/data_occlusion_klab325v2.mat');
data = data.data;
dataSelection = 1:13000;
presIds = unique(data.pres)';
presRows = arrayfun(@(p) find(data.pres == p, 1), presIds);

factory = FeatureProviderFactory('data/features/klab325_orig/', ...
    'data/features/data_occlusion_klab325v2/', ...
    data.pres, dataSelection);
t0Extractor = factory.get(BipolarFeatures(0, AlexnetFc7Features()));
t0 = t0Extractor.extractFeatures(presRows, RunType.Train, []);
t256Extractor = factory.get(...
    HopFeatures(256, BipolarFeatures(0, AlexnetFc7Features())));
t256 = t256Extractor.extractFeatures(presRows, RunType.Train, []);
assert(all(t0(:) == t256(:)), 'not all t0 and t256 features are equal');
end
