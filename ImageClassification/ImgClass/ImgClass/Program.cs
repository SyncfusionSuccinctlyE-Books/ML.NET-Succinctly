namespace ImgClass
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            //Load sample data
            var imageBytes = File.ReadAllBytes(@"C:\PERSONAL-DATA\Projects\Books\ML.NET Succinctly\ML.NET\sample-images\Broccoli\bowl-of-broccoli-2584307_640.jpg");
            ImageClassification.ModelInput sampleData = new ImageClassification.ModelInput()
            {
                ImageSource = imageBytes,
            };

            //Load model and predict output
            var result = ImageClassification.Predict(sampleData);
            Console.WriteLine($"Category {result.PredictedLabel}");
        }
    }
}