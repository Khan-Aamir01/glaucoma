import { Card, CardContent } from "@/components/ui/card";

interface ResultsCardProps {
  imageUrl: string;
  decision: string; // "Normal", "Suspicious", or "Glaucoma"
  confidence: string; // "High" or "Moderate"
  risk: number;
  explanation: string;
  filename: string;
}

export const ResultsCard = ({
  imageUrl,
  decision,
  confidence,
  risk,
  explanation,
  filename,
}: ResultsCardProps) => {
  // --- Determine status tone based on decision ---
  let status = "";
  let statusColor = "";
  let userMessage = "";

  switch (decision) {
    case "Normal":
      status = "✅ No Glaucoma Detected";
      statusColor = "text-green-500";
      userMessage =
        "AI analysis suggests a healthy optic nerve pattern. Regular eye check-ups are still recommended for overall eye health.";
      break;

    case "Suspicious":
      status = "🟡 Suspicious — Needs Monitoring";
      statusColor = "text-yellow-400";
      userMessage =
        "Some optic nerve features appear atypical. It's advisable to monitor regularly or consult an eye specialist for a preventive check-up.";
      break;

    case "Glaucoma":
      status = "🔴 Possible Glaucoma Indicators";
      statusColor = "text-red-500";
      userMessage =
        "AI system detected optic nerve changes consistent with glaucoma. Please consult an ophthalmologist for a detailed evaluation.";
      break;

    default:
      status = "⚪ Awaiting Analysis";
      statusColor = "text-gray-400";
      userMessage = "Upload an image to view the analysis result.";
  }

  return (
    <section id="result" className="container py-16">
      <Card className="bg-card border-border shadow-medical">
        <CardContent className="p-8">
          <div className="grid md:grid-cols-2 gap-8 items-center">
            {/* --- Left side: Result details --- */}
            <div className="space-y-4">
              <p className="text-muted text-2xl font-bold uppercase tracking-wide text-white">
                Results
              </p>

              <div className="space-y-3">
                {/* Status */}
                <h3 className="text-2xl font-bold">
                  Status: <span className={`${statusColor}`}>{status}</span>
                </h3>

                {/* File info */}
                <p className="text-sm text-gray-400 italic">File: {filename}</p>

                {/* Confidence */}
                <p className="text-lg font-medium">
                  Confidence:{" "}
                  <span
                    className={`${
                      confidence === "High"
                        ? "text-green-600"
                        : confidence === "Moderate"
                        ? "text-yellow-500"
                        : "text-gray-500"
                    } font-semibold transition-colors`}
                  >
                    {confidence}
                  </span>
                </p>

                {/* Risk */}
                <p className="text-lg font-medium">
                  Risk Score:{" "}
                  <span
                    className={`${
                      risk < 0.3
                        ? "text-green-500"
                        : risk < 0.7
                        ? "text-yellow-400"
                        : "text-red-500"
                    } font-semibold transition-colors`}
                  >
                    {(risk * 100).toFixed(1)}%
                  </span>
                </p>

                {/* Explanation */}
                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4 mt-2">
                  <h4 className="text-white font-semibold mb-1">
                    🧠 Model Explanation
                  </h4>
                  <p className="text-gray-300 text-sm leading-relaxed">
                    {explanation}
                  </p>
                </div>

                {/* User message */}
                <div className="mt-3">
                  <p className={`${statusColor} font-medium`}>{userMessage}</p>
                </div>

                {/* AI Insight */}
                <div className="bg-gray-800/40 rounded-lg p-4 mt-4">
                  <h4 className="text-white font-semibold mb-2">
                    🔍 AI Insight
                  </h4>
                  <p className="text-gray-300 text-sm leading-relaxed">
                    This result was generated using a deep learning pipeline
                    combining{" "}
                    <span className="font-semibold text-blue-400">ResNet</span>{" "}
                    (for disease classification),{" "}
                    <span className="font-semibold text-blue-400">YOLO</span>{" "}
                    (for optic disc localization), and{" "}
                    <span className="font-semibold text-blue-400">UNet</span>{" "}
                    (for vessel segmentation). These models collectively analyze
                    optic nerve patterns to assist in early glaucoma screening.
                  </p>
                </div>
              </div>
            </div>

            {/* --- Right side: Uploaded Image --- */}
            <div className="relative">
              <div className="aspect-video rounded-xl overflow-hidden bg-gradient-subtle border border-gray-700">
                <img
                  src={imageUrl}
                  alt="Uploaded retinal image"
                  className="w-full h-full object-cover"
                />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </section>
  );
};
