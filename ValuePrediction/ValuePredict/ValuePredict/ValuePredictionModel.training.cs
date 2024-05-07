using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML;

namespace ValuePredict
{
    public partial class ValuePredictionModel
    {
        /// <summary>
        /// Retrains model using the pipeline generated as part of the training process. For more information on how to load data, see aka.ms/loaddata.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="trainData"></param>
        /// <returns></returns>
        public static ITransformer RetrainPipeline(MLContext mlContext, IDataView trainData)
        {
            var pipeline = BuildPipeline(mlContext);
            var model = pipeline.Fit(trainData);

            return model;
        }

        /// <summary>
        /// build the pipeline that is used from model builder. Use this function to retrain model.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <returns></returns>
        public static IEstimator<ITransformer> BuildPipeline(MLContext mlContext)
        {
            // Data process configuration with pipeline data transformations
            var pipeline = mlContext.Transforms.ReplaceMissingValues(
                new []{
                    new InputOutputColumnPair(@"CRIM", @"CRIM"),
                    new InputOutputColumnPair(@"ZN", @"ZN"),
                    new InputOutputColumnPair(@"INDUS", @"INDUS"),
                    new InputOutputColumnPair(@"CHAS", @"CHAS"),
                    new InputOutputColumnPair(@"NOX", @"NOX"),
                    new InputOutputColumnPair(@"RM", @"RM"),
                    new InputOutputColumnPair(@"AGE", @"AGE"),
                    new InputOutputColumnPair(@"DIS", @"DIS"),
                    new InputOutputColumnPair(@"RAD", @"RAD"),
                    new InputOutputColumnPair(@"TAX", @"TAX"),
                    new InputOutputColumnPair(@"PTRATIO", @"PTRATIO"),
                    new InputOutputColumnPair(@"B", @"B"),
                    new InputOutputColumnPair(@"LSAT", @"LSAT")})      
                .Append(mlContext.Transforms.Concatenate(@"Features", 
                    new [] {
                        @"CRIM",@"ZN",@"INDUS",@"CHAS",@"NOX",@"RM",
                        @"AGE",@"DIS",@"RAD",@"TAX",@"PTRATIO",@"B",@"LSAT"
                    })
                )
                .Append(mlContext.Regression.Trainers.LightGbm(
                    new LightGbmRegressionTrainer.Options() {
                            NumberOfLeaves=4,
                            NumberOfIterations=2632,
                            MinimumExampleCountPerLeaf=20,
                            LearningRate=0.599531696349256,
                            LabelColumnName=@"MEDV",
                            FeatureColumnName=@"Features",
                            ExampleWeightColumnName=null,
                            Booster=new GradientBooster.Options() {
                                SubsampleFraction=0.999999776672986,
                                FeatureFraction=0.878434202775917,
                                L1Regularization=2.91194636064836E-10,
                                L2Regularization=0.999999776672986
                            },
                            MaximumBinCountPerFeature=246}
                        )
                );

            return pipeline;
        }
    }
}
