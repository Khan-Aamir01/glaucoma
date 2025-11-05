import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

const faqs = [
  {
    question: "What is Glaucoma?",
    answer: "Glaucoma is a group of eye conditions that damage the optic nerve, often due to high eye pressure. It's a leading cause of blindness worldwide, but early detection and treatment can help preserve vision."
  },
  {
    question: "How accurate is the AI?",
    answer: "Our AI system achieves over 95% accuracy in detecting glaucoma risk indicators. However, this tool is designed to assist healthcare professionals and should not replace comprehensive eye examinations by qualified ophthalmologists."
  },
  {
    question: "Is my data secure?",
    answer: "Yes, we use industry-standard encryption and security protocols. Your medical images are processed securely and are not stored permanently on our servers. All data handling complies with HIPAA regulations."
  },
  {
    question: "What should I do if high risk is detected?",
    answer: "If our analysis indicates high glaucoma risk, please schedule an appointment with an ophthalmologist immediately for comprehensive evaluation and potential treatment options."
  }
];

export const FAQSection = () => {
  return (
    <section id="faq" className="container py-20">
      <div className="space-y-12 max-w-3xl mx-auto">
        <h2 className="text-3xl font-bold font-manrope text-center">FAQ</h2>
        
        <Accordion type="single" collapsible className="space-y-4">
          {faqs.map((faq, index) => (
            <AccordionItem 
              key={index} 
              value={`item-${index}`}
              className="bg-card border-border rounded-xl px-6 py-2"
            >
              <AccordionTrigger className="text-left hover:text-primary transition-medical">
                {faq.question}
              </AccordionTrigger>
              <AccordionContent className="text-muted-foreground leading-relaxed text-gray-100">
                {faq.answer}
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </div>
    </section>
  );
};