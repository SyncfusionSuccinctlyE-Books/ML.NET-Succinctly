using Microsoft.ML;
using Microsoft.ML.Data;

namespace ImgClass
{
    public class ImageClassification
    {
        public class ModelInput
        {
            [LoadColumn(0)]
            [ColumnName(@"Label")]
            public string Label { get; set; }

            [LoadColumn(1)]
            [ColumnName(@"ImageSource")]
            public byte[] ImageSource { get; set; }

        }

        public class ModelOutput
        {
            [ColumnName(@"Label")]
            public uint Label { get; set; }

            [ColumnName(@"ImageSource")]
            public byte[] ImageSource { get; set; }

            [ColumnName(@"PredictedLabel")]
            public string PredictedLabel { get; set; }

            [ColumnName(@"Score")]
            public float[] Score { get; set; }

        }

        private static string MLNetModelPath = Path.GetFullPath("ImgClass.zip");

        public static readonly Lazy<PredictionEngine<ModelInput, ModelOutput>> PredictEngine = new Lazy<PredictionEngine<ModelInput, ModelOutput>>(() => CreatePredictEngine(), true);

        private static PredictionEngine<ModelInput, ModelOutput> CreatePredictEngine()
        {
            var mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load(MLNetModelPath, out var _);
            return mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);
        }

        public static ModelOutput Predict(ModelInput input)
        {
            var predEngine = PredictEngine.Value;
            return predEngine.Predict(input);
        }

        public static IEstimator<ITransformer> BuildPipeline(MLContext mlContext)
        {
            // Data process configuration with pipeline data transformations
            var pipeline = mlContext.Transforms.Conversion.
                MapValueToKey(
                    outputColumnName: @"Label", inputColumnName: @"Label")
            .Append(mlContext.MulticlassClassification.
                Trainers.ImageClassification(
                    labelColumnName: @"Label", 
                    scoreColumnName: @"Score", 
                    featureColumnName: @"ImageSource")
            )
            .Append(mlContext.Transforms.Conversion.
                MapKeyToValue(outputColumnName: @"PredictedLabel", 
                inputColumnName: @"PredictedLabel")
            );

            return pipeline;
        }
    }
}
