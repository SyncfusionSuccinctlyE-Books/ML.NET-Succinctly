//Load sample data
using ValuePredict;

internal class Program
{
    private static void Main(string[] args)
    {
        var sampleData = new ValuePredictionModel.ModelInput()
        {
            CRIM = 0.02731F,
            ZN = 0F,
            INDUS = 7.07F,
            CHAS = 0F,
            NOX = 0.469F,
            RM = 6.421F,
            AGE = 78.9F,
            DIS = 4.9671F,
            RAD = 2F,
            TAX = 242F,
            PTRATIO = 17.8F,
            B = 396.9F,
            LSAT = 9.14F,
        };

        //Load model and predict output
        var result = ValuePredictionModel.Predict(sampleData);
        Console.WriteLine(result.Score.ToString());
    }
}