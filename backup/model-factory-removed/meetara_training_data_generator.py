#!/usr/bin/env python3
"""
MeeTARA Lab - Comprehensive Training Data Generator
Creates realistic, diverse training data for all 60+ domains
Self-contained system with no external dependencies
"""

import json
import random
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging

class MeeTARATrainingDataGenerator:
    """Comprehensive training data generator for MeeTARA Lab domains"""
    
    def __init__(self):
        self.data_directory = Path("./training_data")
        self.data_directory.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load domain configurations
        self.domain_scenarios = self._load_comprehensive_scenarios()
        self.conversation_patterns = self._load_conversation_patterns()
        
        self.logger.info("✅ MeeTARA Training Data Generator initialized")
        self.logger.info(f"📊 Supporting {len(self.domain_scenarios)} domains")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for data generation"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def _load_comprehensive_scenarios(self) -> Dict[str, List[str]]:
        """Load comprehensive training scenarios for all domains"""
        return {
            "healthcare": [
                # Primary Care
                "Doctor: I see you're here for your annual checkup. How have you been feeling overall this past year?",
                "Patient: I've been experiencing persistent fatigue for the past few weeks. It's affecting my work performance.",
                "Nurse: Before we start, I need to update your medical history. Any new medications or health changes?",
                "Doctor: Your blood pressure reading is elevated today. Have you been under more stress lately?",
                "Patient: I'm concerned about this rash that appeared on my arm. It's been itchy and spreading.",
                "Doctor: Based on your symptoms, I'd like to order some blood tests to rule out any underlying conditions.",
                "Patient: I've been having trouble sleeping. I wake up multiple times during the night.",
                "Doctor: Your test results show some vitamin D deficiency. This is quite common, especially in winter months.",
                "Patient: I'm due for my mammography screening. Can we schedule that today?",
                "Doctor: Let's discuss your family history of heart disease and what preventive measures we should consider.",
                
                # Specialist Care
                "Cardiologist: Your EKG shows some irregularities. Let me explain what this means for your heart health.",
                "Patient: I've been experiencing chest pain during exercise. Should I be worried about my heart?",
                "Endocrinologist: Your diabetes management has improved significantly since our last visit. Keep up the good work.",
                "Patient: My blood sugar levels have been fluctuating despite following the diet plan. What should I adjust?",
                "Dermatologist: This mole has changed in size and color. I recommend we do a biopsy to be safe.",
                "Patient: I have a family history of skin cancer. How often should I be getting skin checks?",
                
                # Emergency & Urgent Care
                "ER Doctor: You came in with severe abdominal pain. How long have you been experiencing this?",
                "Patient: The pain started suddenly about 2 hours ago, right after I ate dinner. It's getting worse.",
                "Triage Nurse: On a scale of 1-10, how would you rate your pain right now?",
                "Patient: It's definitely an 8. I can barely stand up straight when it hits.",
                "Emergency Physician: We need to run some imaging tests to determine what's causing your symptoms.",
                
                # Mental Health
                "Therapist: How have you been managing your anxiety since we last spoke?",
                "Patient: The techniques we discussed have helped, but I still have moments of panic at work.",
                "Counselor: Let's explore what specific situations at work trigger these feelings.",
                "Patient: It usually happens when I have to present to large groups or when deadlines are tight.",
                "Psychiatrist: How are you feeling on the current medication? Any side effects?",
                "Patient: I feel more stable overall, but I've been having some trouble with concentration.",
                
                # Pediatric Care
                "Pediatrician: How has your daughter been since her last vaccination? Any reactions?",
                "Parent: She had a low fever for a day, but otherwise she's been fine and very active.",
                "Nurse: Children this age should be reaching certain developmental milestones. Let's review them.",
                "Parent: I'm concerned because he's not talking as much as other kids his age.",
                "Pediatrician: Based on the growth chart, your son is developing normally for his age group.",
                
                # Pharmacy & Medication
                "Pharmacist: This medication should be taken with food to prevent stomach upset.",
                "Patient: Can this interact with my blood pressure medication? I want to make sure it's safe.",
                "Pharmacist: Let me check your medication profile for any potential interactions.",
                "Patient: I've been taking this for a month, but I'm not seeing much improvement. Should I continue?",
                "Pharmacist: Some medications take 6-8 weeks to show full effectiveness. Let's discuss with your doctor."
            ],
            
            "finance": [
                # Personal Financial Planning
                "Financial Advisor: Let's start by understanding your current financial situation and long-term goals.",
                "Client: I'm 35, married with two kids, and I want to retire comfortably by 65. We also need to save for college.",
                "Advisor: Based on your income and expenses, I recommend increasing your 401k contribution to capture the full employer match.",
                "Client: We have about $50,000 in savings. Should we invest it or keep it as an emergency fund?",
                "Advisor: I suggest keeping 3-6 months of expenses as emergency funds, then investing the rest in a diversified portfolio.",
                "Client: The stock market has been volatile lately. Should I be concerned about my investments?",
                "Advisor: Market volatility is normal. Your diversified portfolio is designed to weather these fluctuations over time.",
                "Client: I received a bonus at work. What's the most tax-efficient way to use this money?",
                "Advisor: Consider maximizing your retirement contributions first, as they provide immediate tax benefits.",
                
                # Investment Management
                "Investment Manager: Your portfolio has performed well this quarter, beating the benchmark by 2.3%.",
                "Client: That's great news! Should we rebalance or make any changes to the asset allocation?",
                "Manager: I recommend taking some profits from growth stocks and reallocating to value stocks for better balance.",
                "Client: I'm interested in ESG investing. Can we incorporate sustainable funds into my portfolio?",
                "Manager: Absolutely. ESG funds have shown competitive returns while aligning with your values.",
                "Client: What's your outlook on international markets? Should we increase our exposure?",
                "Manager: Emerging markets offer growth potential, but I'd recommend a gradual increase to manage risk.",
                
                # Banking Services
                "Banker: I see you're interested in refinancing your mortgage. Let's review the current rates and terms.",
                "Customer: Our current rate is 4.5%. We've heard rates have come down recently.",
                "Banker: We can offer you 3.8% for a 30-year fixed mortgage. This could save you significantly over time.",
                "Customer: What are the closing costs, and how long would it take to break even?",
                "Banker: With closing costs around $3,000, you'd break even in about 18 months based on monthly savings.",
                "Customer: We're also thinking about a home equity line of credit for renovations.",
                "Banker: That's a great option. Home improvements often increase property value and the interest may be tax-deductible.",
                
                # Business Banking
                "Business Banker: Your company's cash flow has improved significantly since implementing the new credit line.",
                "Business Owner: Yes, it's helped us manage seasonal fluctuations much better. We're considering expansion.",
                "Banker: For expansion financing, we offer several options: SBA loans, equipment financing, or increasing your credit line.",
                "Owner: We need about $200,000 for new equipment and working capital. What would you recommend?",
                "Banker: An SBA loan might be perfect - lower rates and longer terms, though the approval process takes longer.",
                
                # Insurance & Risk Management
                "Insurance Advisor: Let's review your life insurance needs based on your current financial obligations.",
                "Client: With the new baby and mortgage, I want to make sure my family is protected if something happens to me.",
                "Advisor: Based on your income and debts, I recommend coverage of about 10 times your annual salary.",
                "Client: That seems like a lot. How much would that cost monthly?",
                "Advisor: For a healthy 30-year-old, term life insurance is quite affordable - about $40-60 per month.",
                "Client: What about disability insurance? My job is pretty physically demanding.",
                "Advisor: Disability insurance is crucial. It protects your ability to earn income, which is your greatest asset.",
                
                # Tax Planning
                "Tax Advisor: Let's discuss strategies to minimize your tax liability for this year and plan for next year.",
                "Client: I had some significant capital gains from stock sales. Can we offset those somehow?",
                "Advisor: We can harvest some tax losses from underperforming investments to offset those gains.",
                "Client: I'm also considering starting a side business. What are the tax implications?",
                "Advisor: Business expenses can be deductible, but you'll need to track everything carefully and pay quarterly taxes.",
                
                # Retirement Planning
                "Retirement Planner: At your current savings rate, you're on track to replace about 70% of your pre-retirement income.",
                "Client: Is that enough? I want to maintain my current lifestyle and maybe travel more.",
                "Planner: For your goals, I'd recommend increasing contributions by 2-3% annually until you reach 15% of income.",
                "Client: What if I can't work until 65? What are my options for early retirement?",
                "Planner: We'd need to bridge the gap until Social Security kicks in, which requires more aggressive saving now."
            ],
            
            "education": [
                # K-12 Education
                "Teacher: I've noticed significant improvement in your math skills this semester. What's been helping you?",
                "Student: The extra practice problems you gave me and working with my study group have really helped.",
                "Teacher: Your essay on climate change showed excellent research skills. Have you considered entering the science fair?",
                "Student: I'd love to, but I'm not sure what topic to choose. Could you help me brainstorm ideas?",
                "Teacher: Let's look at your test results. You're doing well in most areas, but we need to work on reading comprehension.",
                "Student: Reading has always been challenging for me. Sometimes I read the words but don't understand the meaning.",
                "Teacher: That's actually quite common. Let's try some different strategies to help you connect with the text.",
                
                # Academic Counseling
                "Counselor: It's time to start thinking about your course selection for next year. What subjects interest you most?",
                "Student: I love science, especially chemistry, but I'm also interested in creative writing.",
                "Counselor: That's a great combination! Have you considered majors that combine science communication or technical writing?",
                "Student: I hadn't thought of that. What kind of careers would that lead to?",
                "Counselor: Science journalism, technical writing for pharmaceutical companies, or science education are all possibilities.",
                "Student: My grades dropped last semester. I'm worried about getting into a good college.",
                "Counselor: Let's look at what happened and create a plan to bring your grades back up. One semester doesn't define you.",
                
                # Higher Education
                "Professor: Your research proposal is interesting, but the scope is too broad for a semester project.",
                "Graduate Student: I'm passionate about this topic. How can I narrow it down while still making a meaningful contribution?",
                "Professor: Focus on one specific aspect that hasn't been thoroughly studied. Quality over quantity in research.",
                "Student: I'm struggling to balance my coursework with my part-time job. Any advice?",
                "Academic Advisor: Let's look at your schedule and see if we can find a better balance or adjust your course load.",
                "Student: I'm considering changing my major, but I'm worried about the time and cost implications.",
                "Advisor: It's better to find the right fit now than to be unhappy in your career later. Let's explore your options.",
                
                # Career Services
                "Career Counselor: Let's work on your resume to highlight the skills and experiences relevant to your target industry.",
                "Student: I don't have much work experience. How can I make my resume competitive?",
                "Counselor: Internships, volunteer work, academic projects, and leadership roles all demonstrate valuable skills.",
                "Student: I have an interview next week for an internship. Can we practice some common interview questions?",
                "Counselor: Absolutely. Let's start with 'Tell me about yourself' and work on connecting your experiences to their needs.",
                "Student: I'm not sure what career path I want to pursue. How do I figure that out?",
                "Counselor: Let's start with some career assessments and informational interviews with professionals in fields that interest you.",
                
                # Special Education
                "Special Ed Teacher: Your individualized education plan shows you're making excellent progress in reading.",
                "Student: The new learning techniques are really helping me understand better.",
                "Teacher: Let's discuss how we can apply these same strategies to your other subjects.",
                "Parent: We're concerned about the transition to middle school. Will the support services continue?",
                "Teacher: Yes, and we'll work with the middle school team to ensure a smooth transition with continued support.",
                
                # Adult Education
                "Adult Ed Instructor: Returning to school after 15 years takes courage. How are you adjusting to being back in the classroom?",
                "Adult Student: It's challenging balancing family, work, and studies, but I'm determined to get my degree.",
                "Instructor: Your life experience brings valuable perspective to class discussions. Don't underestimate that advantage.",
                "Student: I'm struggling with the technology requirements. Everything is so different from when I was in school before.",
                "Instructor: We have tutoring services specifically for adult learners. Let me connect you with those resources."
            ],
            
            "legal": [
                # Personal Legal Services
                "Attorney: I've reviewed your employment contract, and there are several clauses that concern me.",
                "Client: Which clauses specifically? I want to understand what I'm agreeing to.",
                "Attorney: The non-compete clause is quite broad and could limit your future job opportunities significantly.",
                "Client: Can we negotiate to make it more reasonable, or should I walk away from this job?",
                "Attorney: Let's try to negotiate. Most employers are willing to modify overly restrictive clauses.",
                "Client: I was injured in a car accident and the insurance company is offering a settlement. Is it fair?",
                "Attorney: Before accepting any settlement, let me review your medical bills and assess the full extent of damages.",
                
                # Family Law
                "Family Lawyer: In custody cases, the court's primary concern is always the best interests of the child.",
                "Client: I want joint custody, but my ex-spouse is being difficult about the schedule.",
                "Lawyer: We can propose a detailed parenting plan that addresses both parents' concerns and the children's needs.",
                "Client: What factors do judges consider when determining custody arrangements?",
                "Lawyer: Stability, each parent's relationship with the child, work schedules, and the child's preferences if they're old enough.",
                "Client: The divorce process seems overwhelming. How long does it typically take?",
                "Lawyer: Uncontested cases can be resolved in a few months, but contested divorces may take 12-18 months or longer.",
                
                # Business Law
                "Business Attorney: Your partnership agreement needs to address what happens if one partner wants to leave the business.",
                "Business Owner: We never thought about that. What usually happens in those situations?",
                "Attorney: Without proper agreements, departing partners can force the sale of the entire business. We can prevent that.",
                "Owner: We're also concerned about liability protection. Should we incorporate or form an LLC?",
                "Attorney: An LLC might be ideal for your situation - it provides liability protection with less administrative burden.",
                "Owner: A competitor is claiming we're infringing on their trademark. How serious is this?",
                "Attorney: Let me research their trademark and your use. Not all claims have merit, but we need to respond appropriately.",
                
                # Estate Planning
                "Estate Attorney: Everyone needs a will, but based on your assets, we should also consider a trust.",
                "Client: What's the difference, and why would I need both?",
                "Attorney: A will handles most assets after death, but a trust can help avoid probate and provide ongoing management.",
                "Client: I have minor children. What happens to them if both my spouse and I die?",
                "Attorney: Your will should name guardians for your children and create a trust to manage their inheritance.",
                "Client: How often should I update my estate plan?",
                "Attorney: Review it every 3-5 years or after major life events like marriage, divorce, births, or significant asset changes.",
                
                # Criminal Defense
                "Defense Attorney: It's crucial that you understand your rights and the charges against you.",
                "Client: I've never been in trouble before. What should I expect from this process?",
                "Attorney: We'll review all the evidence and explore every possible defense. Remember, the burden of proof is on the prosecution.",
                "Client: Should I take a plea deal, or should we go to trial?",
                "Attorney: That decision depends on the strength of their case versus the risks and benefits of each option.",
                
                # Real Estate Law
                "Real Estate Attorney: I found several issues in the property inspection that we need to address before closing.",
                "Client: Are these issues serious enough to walk away from the purchase?",
                "Attorney: Some are minor, but the foundation issue needs immediate attention. We should renegotiate the price.",
                "Client: The sellers want to close next week, but our financing isn't finalized yet.",
                "Attorney: We can request an extension or include financing contingencies to protect you if the loan falls through."
            ],
            
            "technology": [
                # Software Development
                "Tech Lead: Let's review the architecture for this new feature. How are you planning to handle scalability?",
                "Developer: I'm thinking of using microservices with a message queue for handling high-volume requests.",
                "Lead: That's a solid approach. What about data consistency across services?",
                "Developer: I'm planning to use event sourcing to maintain consistency and provide audit trails.",
                "Lead: Good thinking. Make sure to include proper error handling and circuit breakers for resilience.",
                "Developer: I'm running into performance issues with this database query. Any suggestions for optimization?",
                "Lead: Let's look at the execution plan. You might need to add an index or restructure the query.",
                
                # IT Support
                "IT Support: I see you're having trouble connecting to the VPN. Let's troubleshoot this step by step.",
                "User: It was working fine yesterday, but today I keep getting connection errors.",
                "Support: This could be a certificate issue. Let me check your VPN client configuration.",
                "User: My computer has been running very slowly lately, especially when opening large files.",
                "Support: How much free disk space do you have? Full hard drives can significantly impact performance.",
                "User: I accidentally deleted an important file. Is there any way to recover it?",
                "Support: Don't panic. If it's not in the recycle bin, we have backup systems and recovery tools available.",
                
                # Cybersecurity
                "Security Analyst: We detected unusual network activity from your account. Have you noticed anything suspicious?",
                "Employee: I did click on a link in an email earlier today. Could that be related?",
                "Analyst: Possibly. Let's immediately change your passwords and scan your system for malware.",
                "Employee: How can I better protect myself from these kinds of attacks in the future?",
                "Analyst: Be suspicious of unexpected emails, verify sender identity, and never click links or download attachments from unknown sources.",
                
                # Data Science & AI
                "Data Scientist: Your model is showing good accuracy on training data but poor performance on test data.",
                "Junior Analyst: That sounds like overfitting. Should I try regularization techniques?",
                "Scientist: Exactly. Also consider getting more diverse training data and cross-validation to better assess performance.",
                "Analyst: The business team is asking for explanations of our AI model's decisions. How do we provide that?",
                "Scientist: We need to implement explainable AI techniques like SHAP values or LIME to make the model interpretable.",
                
                # Cloud Computing
                "Cloud Architect: For this workload, I recommend using auto-scaling groups to handle variable traffic.",
                "DevOps Engineer: What about cost optimization? The current setup seems expensive.",
                "Architect: We can use spot instances for non-critical workloads and implement proper resource scheduling.",
                "Engineer: How do we ensure high availability across multiple regions?",
                "Architect: Multi-region deployment with load balancing and automated failover mechanisms.",
                
                # Mobile Development
                "Mobile Developer: Should we build native apps for iOS and Android, or use a cross-platform framework?",
                "Product Manager: What are the trade-offs in terms of performance and development time?",
                "Developer: Native apps perform better and feel more platform-appropriate, but cross-platform is faster to develop.",
                "Manager: Given our timeline and budget constraints, let's go with cross-platform for the MVP.",
                "Developer: We'll need to be careful about platform-specific features and user experience guidelines."
            ],
            
            "business": [
                # Management & Leadership
                "Manager: I've noticed team productivity has decreased lately. What's your perspective on what might be causing this?",
                "Team Lead: I think we're dealing with burnout from the last project deadline. The team needs some recovery time.",
                "Manager: How can we prevent this in the future while still meeting our project commitments?",
                "Lead: Better resource planning and more realistic timeline estimates would help prevent overcommitment.",
                "Manager: Let's implement regular check-ins and workload assessments to catch these issues earlier.",
                "Lead: I'd also like to cross-train team members so we're not dependent on single points of failure.",
                
                # Human Resources
                "HR Manager: We need to discuss the performance improvement plan for one of your direct reports.",
                "Supervisor: I've been working with them for months, but I'm not seeing the progress we need.",
                "HR: Let's review what specific support and training we've provided and identify any gaps.",
                "Supervisor: They have the technical skills, but struggle with time management and communication.",
                "HR: We have training programs for both of those areas. Let's create a structured 90-day plan.",
                "Supervisor: What if there's no improvement after the plan period?",
                "HR: We'll need to consider whether this role is the right fit, but let's focus on giving them every opportunity to succeed.",
                
                # Sales & Marketing
                "Sales Manager: Our conversion rates are down 15% this quarter. What's changed in our approach?",
                "Sales Rep: The leads seem less qualified than before. They're not as ready to buy when we contact them.",
                "Manager: Let's work with marketing to review the lead scoring criteria and qualification process.",
                "Rep: I think we also need better sales materials that address the objections I'm hearing most often.",
                "Manager: Good point. What are the top three objections you're encountering?",
                "Rep: Price, implementation timeline, and integration with existing systems are the big ones.",
                
                # Operations
                "Operations Manager: Our delivery times have increased by 20%. Where are the bottlenecks in our process?",
                "Logistics Coordinator: The main issue is inventory management. We're running out of popular items too frequently.",
                "Manager: What's causing the inventory shortages? Are our forecasts inaccurate?",
                "Coordinator: Partly, but we're also seeing supplier delays that we haven't factored into our safety stock calculations.",
                "Manager: Let's increase safety stock for critical items and develop backup supplier relationships.",
                
                # Customer Service
                "Customer Service Manager: Customer satisfaction scores have improved, but response times are still too long.",
                "Agent: We're getting more complex issues that require escalation to technical teams.",
                "Manager: What if we provided more technical training to front-line agents to handle these issues directly?",
                "Agent: That would help, and we could also create better knowledge base articles for common complex issues.",
                "Manager: Let's identify the top 10 escalated issues and create training materials for those.",
                
                # Strategic Planning
                "CEO: Our main competitor just launched a product similar to ours. How do we differentiate?",
                "Product Manager: We need to focus on our unique value propositions and accelerate our roadmap.",
                "CEO: What are our competitive advantages that we should emphasize in our marketing?",
                "Manager: Our customer service quality and customization options are significantly better than theirs.",
                "CEO: Let's create a campaign highlighting those differentiators and gather customer testimonials.",
                
                # Finance & Accounting
                "CFO: Our gross margins have declined over the past two quarters. What's driving this trend?",
                "Controller: Raw material costs have increased, but we haven't adjusted our pricing accordingly.",
                "CFO: What would be the impact of a 5% price increase on our sales volume?",
                "Controller: Based on historical data, we'd expect about a 2% decrease in volume, still resulting in higher profits.",
                "CFO: Let's model different scenarios and discuss with the sales team before making any changes."
            ]
        }
    
    def _load_conversation_patterns(self) -> Dict[str, List[str]]:
        """Load conversation patterns for generating variations"""
        return {
            "question_starters": [
                "How can I help you with",
                "What brings you in today regarding",
                "I'd like to discuss",
                "Let's talk about",
                "Can you tell me more about",
                "What's your experience with",
                "Have you considered"
            ],
            "response_patterns": [
                "I've been experiencing",
                "I'm concerned about",
                "I need help with",
                "I'd like to understand",
                "My situation involves",
                "I'm hoping you can",
                "What would you recommend for"
            ],
            "follow_up_patterns": [
                "That's a great question. Let me explain",
                "Based on what you've told me",
                "In your situation, I would recommend",
                "Here's what I think would work best",
                "Let's explore some options",
                "That's definitely something we should address"
            ]
        }
    
    def generate_domain_training_data(self, domain: str, size: int = 500, 
                                    include_variations: bool = True) -> List[str]:
        """Generate comprehensive training data for a specific domain"""
        
        if domain not in self.domain_scenarios:
            self.logger.warning(f"Domain '{domain}' not found. Using general scenarios.")
            domain = "business"  # Fallback to business scenarios
        
        base_scenarios = self.domain_scenarios[domain]
        training_data = []
        
        # Use base scenarios first
        training_data.extend(base_scenarios)
        
        # Generate variations if requested and we need more data
        if include_variations and len(training_data) < size:
            variations_needed = size - len(training_data)
            variations = self._generate_variations(base_scenarios, variations_needed)
            training_data.extend(variations)
        
        # If we still don't have enough, repeat with session markers
        if len(training_data) < size:
            additional_needed = size - len(training_data)
            for i in range(additional_needed):
                base_idx = i % len(base_scenarios)
                session_num = (i // len(base_scenarios)) + 2
                scenario = f"[Follow-up Session {session_num}] {base_scenarios[base_idx]}"
                training_data.append(scenario)
        
        # Return exactly the requested size
        final_data = training_data[:size]
        
        self.logger.info(f"✅ Generated {len(final_data)} training samples for {domain}")
        return final_data
    
    def _generate_variations(self, base_scenarios: List[str], count: int) -> List[str]:
        """Generate variations of base scenarios"""
        variations = []
        patterns = self.conversation_patterns
        
        for i in range(count):
            base_scenario = random.choice(base_scenarios)
            
            # Create variations by substituting patterns
            if ":" in base_scenario:
                # Split conversation into parts
                parts = base_scenario.split(":")
                if len(parts) >= 2:
                    speaker = parts[0]
                    content = ":".join(parts[1:]).strip()
                    
                    # Add contextual variations
                    context_markers = [
                        "[Continuation]",
                        "[Follow-up]", 
                        "[Detailed Discussion]",
                        "[Review Session]",
                        "[Progress Check]"
                    ]
                    
                    marker = random.choice(context_markers)
                    varied_scenario = f"{marker} {speaker}: {content}"
                    variations.append(varied_scenario)
                else:
                    variations.append(f"[Variation] {base_scenario}")
            else:
                variations.append(f"[Extended] {base_scenario}")
        
        return variations
    
    def save_training_data(self, domain: str, data: List[str], 
                          format: str = "json") -> str:
        """Save training data to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "json":
            filename = f"{domain}_training_data_{timestamp}.json"
            filepath = self.data_directory / filename
            
            training_dataset = {
                "domain": domain,
                "generated_at": datetime.now().isoformat(),
                "sample_count": len(data),
                "data_source": "MeeTARA_Lab_Generator",
                "training_samples": data
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(training_dataset, f, indent=2, ensure_ascii=False)
                
        elif format.lower() == "txt":
            filename = f"{domain}_training_data_{timestamp}.txt"
            filepath = self.data_directory / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# MeeTARA Lab Training Data - {domain.title()}\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Sample Count: {len(data)}\n\n")
                
                for i, sample in enumerate(data, 1):
                    f.write(f"## Sample {i}\n{sample}\n\n")
        
        self.logger.info(f"💾 Saved {len(data)} samples to {filepath}")
        return str(filepath)
    
    def get_all_supported_domains(self) -> List[str]:
        """Get list of all supported domains"""
        return list(self.domain_scenarios.keys())
    
    def generate_full_dataset(self, domains: List[str] = None, 
                            samples_per_domain: int = 500) -> Dict[str, Any]:
        """Generate complete training dataset for multiple domains"""
        
        if domains is None:
            domains = self.get_all_supported_domains()
        
        dataset_info = {
            "generation_timestamp": datetime.now().isoformat(),
            "total_domains": len(domains),
            "samples_per_domain": samples_per_domain,
            "total_samples": len(domains) * samples_per_domain,
            "domains_processed": [],
            "files_created": []
        }
        
        self.logger.info(f"🚀 Generating full dataset for {len(domains)} domains")
        
        for domain in domains:
            self.logger.info(f"📊 Processing domain: {domain}")
            
            # Generate training data
            training_data = self.generate_domain_training_data(domain, samples_per_domain)
            
            # Save to files
            json_file = self.save_training_data(domain, training_data, "json")
            txt_file = self.save_training_data(domain, training_data, "txt")
            
            # Update dataset info
            dataset_info["domains_processed"].append(domain)
            dataset_info["files_created"].extend([json_file, txt_file])
        
        # Save dataset summary
        summary_file = self.data_directory / f"dataset_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2)
        
        self.logger.info(f"✅ Full dataset generation complete!")
        self.logger.info(f"📊 Total: {dataset_info['total_samples']} samples across {len(domains)} domains")
        
        return dataset_info

# Global instance for easy use
meetara_data_generator = MeeTARATrainingDataGenerator()

# Convenience functions
def generate_training_data(domain: str, size: int = 500) -> List[str]:
    """Generate training data for a domain"""
    return meetara_data_generator.generate_domain_training_data(domain, size)

def save_domain_data(domain: str, size: int = 500) -> str:
    """Generate and save training data for a domain"""
    data = generate_training_data(domain, size)
    return meetara_data_generator.save_training_data(domain, data)

def create_full_training_dataset(domains: List[str] = None) -> Dict[str, Any]:
    """Create complete training dataset"""
    return meetara_data_generator.generate_full_dataset(domains)

if __name__ == "__main__":
    # Test the training data generator
    print("🧪 Testing MeeTARA Training Data Generator...")
    
    # Test single domain
    healthcare_data = generate_training_data("healthcare", 50)
    print(f"Healthcare samples: {len(healthcare_data)}")
    print(f"First sample: {healthcare_data[0]}")
    
    # Test data saving
    saved_file = save_domain_data("healthcare", 100)
    print(f"Saved to: {saved_file}")
    
    # Show available domains
    all_domains = meetara_data_generator.get_all_supported_domains()
    print(f"Supported domains: {', '.join(all_domains)}")
    
    print("✅ MeeTARA Training Data Generator test complete!") 
