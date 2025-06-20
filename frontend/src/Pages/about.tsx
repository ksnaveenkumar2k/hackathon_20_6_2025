
"use client"

import {
  Target,
  Brain,
  Zap,
  TrendingUp,
  CheckCircle,
  Star,
  Award,
  Lightbulb,
  Sparkles,
  Users,
  Shield,
  Rocket,
  Globe,
  ArrowRight,
  Building,
} from "lucide-react"
import Navbar from "../components/navbar"

const About = () => {
  const features = [
    {
      icon: <Brain className="h-10 w-10 text-yellow-600" />,
      title: "AI-Powered Analysis",
      description:
        "Advanced machine learning algorithms analyze your profile against market standards and provide personalized insights tailored to your career goals.",
      highlight: "Smart Analysis",
    },
    {
      icon: <Target className="h-10 w-10 text-yellow-600" />,
      title: "Reality Matching",
      description:
        "Compare your aspirations with actual market conditions, placement statistics, and industry trends for realistic career planning.",
      highlight: "Market Alignment",
    },
    {
      icon: <TrendingUp className="h-10 w-10 text-yellow-600" />,
      title: "Gap Analysis",
      description:
        "Identify specific areas for improvement with detailed gap analysis, scoring, and comprehensive skill assessment reports.",
      highlight: "Skill Assessment",
    },
    {
      icon: <Lightbulb className="h-10 w-10 text-yellow-600" />,
      title: "Actionable Plans",
      description:
        "Get micro-OKRs, step-by-step action plans, and personalized roadmaps to bridge your skill gaps effectively.",
      highlight: "Strategic Planning",
    },
  ]

  const technologies = [
    {
      name: "LangChain",
      description: "Advanced language processing and AI orchestration",
      icon: <Zap className="h-8 w-8 text-black" />,
    },
    {
      name: "Google Generative AI",
      description: "Cutting-edge AI models for intelligent analysis",
      icon: <Brain className="h-8 w-8 text-black" />,
    },
    {
      name: "React & Next.js",
      description: "Modern, responsive web interface",
      icon: <Globe className="h-8 w-8 text-black" />,
    },
    {
      name: "Python & FastAPI",
      description: "Robust backend processing and API services",
      icon: <Rocket className="h-8 w-8 text-black" />,
    },
  ]

  const stats = [
    { number: "10,000+", label: "Students Analyzed", icon: <Users className="h-8 w-8 text-yellow-400" /> },
    { number: "95%", label: "Accuracy Rate", icon: <Target className="h-8 w-8 text-yellow-400" /> },
    { number: "500+", label: "Companies Tracked", icon: <Building className="h-8 w-8 text-yellow-400" /> },
    { number: "24/7", label: "AI Support", icon: <Shield className="h-8 w-8 text-yellow-400" /> },
  ]

  return (
    <>
      <Navbar />
      <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-100">
        {/* Enhanced Hero Section */}
        <div className="bg-gradient-to-r from-black via-gray-900 to-black text-white py-24 lg:py-32 relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-yellow-400/10 to-transparent"></div>
          <div className="absolute top-0 right-0 w-96 h-96 bg-yellow-400 bg-opacity-5 rounded-full -mr-48 -mt-48"></div>
          <div className="absolute bottom-0 left-0 w-64 h-64 bg-yellow-400 bg-opacity-5 rounded-full -ml-32 -mb-32"></div>

          <div className="max-w-7xl mx-auto px-6 relative z-10">
            <div className="text-center">
              <div className="flex justify-center mb-8">
                <div className="bg-yellow-400 p-6 rounded-3xl animate-pulse shadow-2xl">
                  <Target className="h-16 w-16 text-black" />
                </div>
              </div>
              <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold mb-8 animate-fade-in">
                About <span className="text-yellow-400">Reality Matcher</span>
              </h1>
              <p className="text-xl lg:text-2xl text-gray-300 max-w-4xl mx-auto leading-relaxed mb-12 animate-fade-in-up">
                Empowering students with AI-driven insights to bridge the gap between career aspirations and market
                realities through intelligent analysis and personalized guidance
              </p>

              {/* Stats Row */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-8 max-w-4xl mx-auto">
                {stats.map((stat, index) => (
                  <div
                    key={index}
                    className="text-center animate-fade-in-up"
                    style={{ animationDelay: `${index * 100}ms` }}
                  >
                    <div className="flex justify-center mb-3">{stat.icon}</div>
                    <div className="text-2xl lg:text-3xl font-bold text-yellow-400 mb-1">{stat.number}</div>
                    <div className="text-sm lg:text-base text-gray-300">{stat.label}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Enhanced Mission Section */}
        <div className="py-24 lg:py-32">
          <div className="max-w-7xl mx-auto px-6">
            <div className="bg-white rounded-3xl shadow-2xl p-12 lg:p-20 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-40 h-40 bg-yellow-400 bg-opacity-5 rounded-full -mr-20 -mt-20"></div>

              <div className="text-center mb-16 relative z-10">
                <h2 className="text-4xl lg:text-5xl font-bold text-black mb-6">Our Mission</h2>
                <div className="w-32 h-2 bg-gradient-to-r from-yellow-400 to-amber-500 mx-auto mb-8 rounded-full"></div>
                <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                  Transforming career planning through intelligent technology and personalized insights
                </p>
              </div>

              <div className="grid lg:grid-cols-2 gap-16 items-center">
                <div className="space-y-8">
                  <div className="space-y-6">
                    <p className="text-lg lg:text-xl text-gray-700 leading-relaxed">
                      The Reality-Matching Simulator is an AI-powered platform designed to help students align their
                      placement expectations with market realities. By analyzing aspirations and profiles, it provides
                      personalized guidance and actionable plans.
                    </p>
                    <p className="text-lg lg:text-xl text-gray-700 leading-relaxed">
                      Built with cutting-edge technology, including LangChain and Google Generative AI, our tool offers
                      transparent and empathetic coaching to empower your career journey.
                    </p>
                  </div>

                  <div className="grid md:grid-cols-2 gap-6">
                    {[
                      { icon: <CheckCircle className="h-6 w-6 text-green-500" />, text: "Personalized Analysis" },
                      { icon: <CheckCircle className="h-6 w-6 text-green-500" />, text: "Market Insights" },
                      { icon: <CheckCircle className="h-6 w-6 text-green-500" />, text: "Career Guidance" },
                      { icon: <CheckCircle className="h-6 w-6 text-green-500" />, text: "Action Plans" },
                    ].map((item, index) => (
                      <div key={index} className="flex items-center space-x-3">
                        {item.icon}
                        <span className="text-gray-700 font-medium">{item.text}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-gradient-to-br from-yellow-400 via-yellow-500 to-amber-500 p-10 lg:p-12 rounded-3xl text-black relative overflow-hidden shadow-2xl">
                  <div className="absolute top-0 right-0 w-32 h-32 bg-white bg-opacity-10 rounded-full -mr-16 -mt-16"></div>
                  <div className="relative z-10">
                    <div className="text-center mb-8">
                      <Award className="h-20 w-20 mx-auto mb-6" />
                      <h3 className="text-2xl lg:text-3xl font-bold mb-6">Why Choose Us?</h3>
                    </div>
                    <ul className="space-y-4">
                      {[
                        "AI-powered career guidance",
                        "Real-time market analysis",
                        "Actionable improvement plans",
                        "Transparent feedback system",
                        "Personalized learning paths",
                        "Industry expert insights",
                      ].map((item, index) => (
                        <li key={index} className="flex items-center space-x-3">
                          <Star className="h-5 w-5 flex-shrink-0" />
                          <span className="text-lg font-medium">{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Enhanced Features Section */}
        <div className="py-24 lg:py-32 bg-gradient-to-br from-gray-100 to-gray-50">
          <div className="max-w-7xl mx-auto px-6">
            <div className="text-center mb-20">
              <h2 className="text-4xl lg:text-5xl font-bold text-black mb-6">Key Features</h2>
              <div className="w-32 h-2 bg-gradient-to-r from-yellow-400 to-amber-500 mx-auto mb-8 rounded-full"></div>
              <p className="text-xl lg:text-2xl text-gray-600 max-w-3xl mx-auto">
                Discover how our platform helps you achieve your career goals with intelligent insights
              </p>
            </div>

            <div className="grid md:grid-cols-2 xl:grid-cols-4 gap-8 lg:gap-10">
              {features.map((feature, index) => (
                <div
                  key={index}
                  className="bg-white p-8 lg:p-10 rounded-3xl shadow-xl hover:shadow-2xl transition-all duration-500 transform hover:-translate-y-3 group relative overflow-hidden"
                >
                  <div className="absolute top-0 right-0 w-24 h-24 bg-yellow-400 bg-opacity-5 rounded-full -mr-12 -mt-12 group-hover:bg-opacity-10 transition-all duration-300"></div>

                  <div className="text-center relative z-10">
                    <div className="bg-gradient-to-br from-yellow-50 to-amber-50 p-6 rounded-2xl w-fit mx-auto mb-6 group-hover:scale-110 transition-transform duration-300">
                      {feature.icon}
                    </div>

                    <div className="mb-4">
                      <span className="inline-block bg-yellow-100 text-yellow-800 text-xs font-bold px-3 py-1 rounded-full mb-3">
                        {feature.highlight}
                      </span>
                      <h3 className="text-xl lg:text-2xl font-bold text-black mb-4">{feature.title}</h3>
                    </div>

                    <p className="text-gray-600 leading-relaxed text-base lg:text-lg">{feature.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Enhanced Technology Section */}
        <div className="py-24 lg:py-32">
          <div className="max-w-7xl mx-auto px-6">
            <div className="bg-gradient-to-br from-black via-gray-900 to-black rounded-3xl p-12 lg:p-20 text-white relative overflow-hidden shadow-2xl">
              <div className="absolute top-0 right-0 w-40 h-40 bg-yellow-400 bg-opacity-10 rounded-full -mr-20 -mt-20"></div>
              <div className="absolute bottom-0 left-0 w-32 h-32 bg-yellow-400 bg-opacity-5 rounded-full -ml-16 -mb-16"></div>

              <div className="text-center mb-16 relative z-10">
                <h2 className="text-4xl lg:text-5xl font-bold mb-6">
                  Powered by <span className="text-yellow-400">Advanced Technology</span>
                </h2>
                <div className="w-32 h-2 bg-gradient-to-r from-yellow-400 to-amber-500 mx-auto mb-8 rounded-full"></div>
                <p className="text-xl lg:text-2xl text-gray-300 max-w-3xl mx-auto">
                  Built with cutting-edge AI and modern web technologies for optimal performance
                </p>
              </div>

              <div className="grid md:grid-cols-2 xl:grid-cols-4 gap-8 lg:gap-10">
                {technologies.map((tech, index) => (
                  <div
                    key={index}
                    className="text-center p-8 lg:p-10 bg-gray-800 rounded-2xl hover:bg-gray-700 transition-all duration-300 transform hover:scale-105 group relative overflow-hidden"
                  >
                    <div className="absolute inset-0 bg-gradient-to-br from-yellow-400/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>

                    <div className="relative z-10">
                      <div className="bg-yellow-400 p-4 rounded-2xl w-fit mx-auto mb-6 group-hover:scale-110 transition-transform duration-300">
                        {tech.icon}
                      </div>
                      <h3 className="text-xl lg:text-2xl font-bold text-yellow-400 mb-4">{tech.name}</h3>
                      <p className="text-gray-300 text-base lg:text-lg leading-relaxed">{tech.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Enhanced CTA Section */}
        <div className="py-24 lg:py-32 bg-gradient-to-r from-yellow-400 via-yellow-500 to-amber-500 relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>
          <div className="absolute top-0 left-0 w-96 h-96 bg-black bg-opacity-5 rounded-full -ml-48 -mt-48"></div>
          <div className="absolute bottom-0 right-0 w-64 h-64 bg-black bg-opacity-5 rounded-full -mr-32 -mb-32"></div>

          <div className="max-w-5xl mx-auto px-6 text-center relative z-10">
            <div className="mb-8">
              <Sparkles className="h-16 w-16 mx-auto text-black mb-6" />
            </div>
            <h2 className="text-4xl lg:text-5xl font-bold text-black mb-8">Ready to Transform Your Career Journey?</h2>
            <p className="text-xl lg:text-2xl text-black mb-12 opacity-90 max-w-3xl mx-auto leading-relaxed">
              Join thousands of students who have already discovered their true potential and achieved their career
              goals
            </p>
            <a
              href="/"
              className="inline-flex items-center space-x-3 bg-black text-white px-10 py-6 rounded-2xl font-bold text-xl lg:text-2xl hover:bg-gray-800 transition-all duration-300 transform hover:scale-105 shadow-2xl group"
            >
              <Target className="h-6 w-6 group-hover:rotate-12 transition-transform duration-300" />
              <span>Start Your Analysis</span>
              <ArrowRight className="h-6 w-6 group-hover:translate-x-1 transition-transform duration-300" />
            </a>
          </div>
        </div>

        {/* Enhanced Footer */}
        <footer className="bg-gradient-to-r from-black via-gray-900 to-black text-white py-16 lg:py-20">
          <div className="max-w-7xl mx-auto px-6">
            <div className="text-center">
              <div className="flex items-center justify-center space-x-3 mb-6">
                <div className="p-2 bg-yellow-400 rounded-xl">
                  <Target className="h-10 w-10 text-black" />
                </div>
                <span className="text-3xl lg:text-4xl font-bold">Reality Matcher</span>
              </div>
              <p className="text-gray-400 mb-8 text-lg lg:text-xl max-w-2xl mx-auto">
                Empowering careers through AI-driven insights and intelligent career planning
              </p>

              <div className="border-t border-gray-800 pt-8">
                <p className="text-gray-500 text-base lg:text-lg">
                  © {new Date().getFullYear()} Reality-Matching Simulator. All rights reserved.
                </p>
              </div>
            </div>
          </div>
        </footer>
      </div>

      <style jsx>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes fade-in-up {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .animate-fade-in {
          animation: fade-in 0.6s ease-out;
        }

        .animate-fade-in-up {
          animation: fade-in-up 0.8s ease-out;
        }
      `}</style>
    </>
  )
}

export default About
