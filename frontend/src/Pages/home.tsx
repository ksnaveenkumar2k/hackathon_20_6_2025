
"use client"

import type React from "react"
import { useState } from "react"
import {
  DollarSign,
  Briefcase,
  Building,
  Target,
  Upload,
  Linkedin,
  Github,
  Code,
  Loader2,
  CheckCircle,
  AlertCircle,
  TrendingUp,
  Calendar,
  BookOpen,
  XCircle,
  AlertTriangle,
  Award,
  BarChart3,
  Sparkles,
  Zap,
  Brain,
} from "lucide-react"
import Navbar from "../components/navbar"

// Define types for InputField props
interface InputFieldProps {
  type: string
  placeholder: string
  value: string | null
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
  error?: string
  label?: string
  icon?: React.ReactNode
}

// Define types for FileInputField props
interface FileInputFieldProps {
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
  error?: string
  label?: string
  accept?: string
  icon?: React.ReactNode
}

// Reusable InputField component
const InputField: React.FC<InputFieldProps> = ({ type, placeholder, value, onChange, error, label, icon }) => (
  <div className="mb-6 group">
    {label && (
      <label className="flex items-center space-x-2 text-sm font-semibold text-gray-800 mb-3">
        <div className="p-1 bg-yellow-100 rounded-md">{icon}</div>
        <span>{label}</span>
      </label>
    )}
    <div className="relative">
      <input
        type={type}
        placeholder={placeholder}
        value={value || ""}
        onChange={onChange}
        className={`w-full p-4 lg:p-5 border-2 rounded-2xl focus:outline-none transition-all duration-300 text-base lg:text-lg ${
          error
            ? "border-red-400 focus:border-red-500 bg-red-50 shadow-red-100"
            : "border-gray-200 focus:border-yellow-400 bg-white hover:border-yellow-300 focus:shadow-lg focus:shadow-yellow-100"
        } group-hover:shadow-md`}
      />
      {error && (
        <div className="flex items-center space-x-2 mt-3 text-red-600 animate-fade-in">
          <AlertCircle className="h-4 w-4" />
          <p className="text-sm font-medium">{error}</p>
        </div>
      )}
    </div>
  </div>
)

// Reusable FileInputField component
const FileInputField: React.FC<FileInputFieldProps> = ({ onChange, error, label, accept, icon }) => (
  <div className="mb-6 group">
    {label && (
      <label className="flex items-center space-x-2 text-sm font-semibold text-gray-800 mb-3">
        <div className="p-1 bg-yellow-100 rounded-md">{icon}</div>
        <span>{label}</span>
      </label>
    )}
    <div className="relative">
      <input
        type="file"
        accept={accept}
        onChange={onChange}
        className={`w-full p-4 lg:p-5 border-2 rounded-2xl focus:outline-none transition-all duration-300 text-base lg:text-lg file:mr-4 file:py-3 file:px-6 file:rounded-xl file:border-0 file:text-sm file:font-bold file:bg-yellow-400 file:text-black hover:file:bg-yellow-500 file:transition-all file:duration-200 ${
          error
            ? "border-red-400 focus:border-red-500 bg-red-50"
            : "border-gray-200 focus:border-yellow-400 bg-white hover:border-yellow-300 focus:shadow-lg focus:shadow-yellow-100"
        } group-hover:shadow-md`}
      />
      {error && (
        <div className="flex items-center space-x-2 mt-3 text-red-600 animate-fade-in">
          <AlertCircle className="h-4 w-4" />
          <p className="text-sm font-medium">{error}</p>
        </div>
      )}
    </div>
  </div>
)

// Define state types
interface Aspirations {
  ctc: string
  role: string
  companies: string
  okrs: string
}

interface Profile {
  resume: File | null
  linkedin: string
  github: string
  leetcode: string
}

interface Result {
  gapScore: number
  gapDetails: { [key: string]: string }
  microOkrs: { task: string; resource: string; timeline: string }[]
  report: { summary: string; feedback: string; next_steps: string }
}

interface Errors {
  ctc?: string
  role?: string
  companies?: string
  okrs?: string
  profile?: string
  linkedin?: string
  github?: string
  leetcode?: string
}

// Enhanced Gap Analysis Item Component
interface GapAnalysisItemProps {
  title: string
  data: string
  icon: React.ReactNode
}

const GapAnalysisItem: React.FC<GapAnalysisItemProps> = ({ title, data, icon }) => {
  // Parse the JSON-like string to extract alignment and description
  const parseGapData = (dataString: string) => {
    try {
      let parsed
      if (typeof dataString === "string") {
        if (dataString.startsWith("{") && dataString.endsWith("}")) {
          parsed = JSON.parse(dataString)
        } else {
          return { alignment: "Unknown", description: dataString }
        }
      } else {
        parsed = dataString
      }

      return {
        alignment: parsed.alignment || "Unknown",
        description: parsed.description || "No description available",
      }
    } catch (error) {
      const alignmentMatch = dataString.match(/"alignment":\s*"([^"]*)"/)
      const descriptionMatch = dataString.match(/"description":\s*"([^"]*)"/)

      return {
        alignment: alignmentMatch ? alignmentMatch[1] : "Unknown",
        description: descriptionMatch ? descriptionMatch[1] : dataString,
      }
    }
  }

  const { alignment, description } = parseGapData(data)

  // Get alignment styling and icon
  const getAlignmentStyle = (alignment: string) => {
    const normalizedAlignment = alignment.toLowerCase()
    switch (normalizedAlignment) {
      case "good":
      case "excellent":
        return {
          bgColor: "bg-gradient-to-br from-green-50 to-emerald-50",
          borderColor: "border-green-200",
          textColor: "text-green-800",
          badgeColor: "bg-green-100 text-green-800 border border-green-200",
          icon: <CheckCircle className="h-5 w-5 text-green-600" />,
          progressColor: "bg-gradient-to-r from-green-400 to-emerald-500",
          progressValue: 85,
          glowColor: "shadow-green-100",
        }
      case "moderate":
      case "average":
        return {
          bgColor: "bg-gradient-to-br from-yellow-50 to-amber-50",
          borderColor: "border-yellow-200",
          textColor: "text-yellow-800",
          badgeColor: "bg-yellow-100 text-yellow-800 border border-yellow-200",
          icon: <AlertTriangle className="h-5 w-5 text-yellow-600" />,
          progressColor: "bg-gradient-to-r from-yellow-400 to-amber-500",
          progressValue: 60,
          glowColor: "shadow-yellow-100",
        }
      case "poor":
      case "low":
      case "needs improvement":
        return {
          bgColor: "bg-gradient-to-br from-red-50 to-rose-50",
          borderColor: "border-red-200",
          textColor: "text-red-800",
          badgeColor: "bg-red-100 text-red-800 border border-red-200",
          icon: <XCircle className="h-5 w-5 text-red-600" />,
          progressColor: "bg-gradient-to-r from-red-400 to-rose-500",
          progressValue: 30,
          glowColor: "shadow-red-100",
        }
      default:
        return {
          bgColor: "bg-gradient-to-br from-gray-50 to-slate-50",
          borderColor: "border-gray-200",
          textColor: "text-gray-800",
          badgeColor: "bg-gray-100 text-gray-800 border border-gray-200",
          icon: <AlertCircle className="h-5 w-5 text-gray-600" />,
          progressColor: "bg-gradient-to-r from-gray-400 to-slate-500",
          progressValue: 50,
          glowColor: "shadow-gray-100",
        }
    }
  }

  const style = getAlignmentStyle(alignment)

  return (
    <div
      className={`${style.bgColor} ${style.borderColor} border-2 rounded-2xl p-6 lg:p-8 transition-all duration-500 hover:shadow-xl hover:scale-105 ${style.glowColor} animate-fade-in-up`}
    >
      <div className="flex items-start justify-between mb-6">
        <div className="flex items-center space-x-4">
          <div className="p-3 bg-white rounded-xl shadow-sm border border-gray-100">{icon}</div>
          <div>
            <h4 className="text-lg lg:text-xl font-bold text-black mb-2">{title}</h4>
            <div
              className={`inline-flex items-center space-x-2 px-4 py-2 rounded-full text-sm font-bold ${style.badgeColor}`}
            >
              {style.icon}
              <span>{alignment}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced Progress Bar */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-3">
          <span className="text-sm font-semibold text-gray-700">Alignment Score</span>
          <span className="text-lg font-bold text-gray-900">{style.progressValue}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
          <div
            className={`${style.progressColor} h-3 rounded-full transition-all duration-1000 ease-out shadow-sm`}
            style={{ width: `${style.progressValue}%` }}
          ></div>
        </div>
      </div>

      {/* Description */}
      <p className={`${style.textColor} leading-relaxed text-base lg:text-lg`}>{description}</p>
    </div>
  )
}

const Home: React.FC = () => {
  const [aspirations, setAspirations] = useState<Aspirations>({ ctc: "", role: "", companies: "", okrs: "" })
  const [profile, setProfile] = useState<Profile>({ resume: null, linkedin: "", github: "", leetcode: "" })
  const [errors, setErrors] = useState<Errors>({})
  const [result, setResult] = useState<Result | null>(null)
  const [loading, setLoading] = useState(false)

  // Handle input changes
  const handleInputChange = (field: keyof Aspirations | keyof Profile, value: string | File) => {
    if (["ctc", "role", "companies", "okrs"].includes(field as string)) {
      setAspirations({ ...aspirations, [field]: value as string })
    } else {
      setProfile({ ...profile, [field]: value })
    }
    const error = validateField(field as string, value as string)
    setErrors((prev) => ({ ...prev, [field]: error }))
  }

  // Validate individual field
  const validateField = (name: string, value: string): string | undefined => {
    if (!value && ["ctc", "role"].includes(name)) return `${name.toUpperCase()} is required`
    const urlPattern = /^(https?:\/\/)?([\da-z.-]+)\.([a-z.]{2,6})([/\w .-]*)*\/?$/
    if (["linkedin", "github", "leetcode"].includes(name) && value && !urlPattern.test(value)) {
      return `Invalid ${name} URL`
    }
    return undefined
  }

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const validationErrors = Object.fromEntries(
      Object.keys({ ...aspirations, ...profile }).map((field) => [
        field,
        validateField(
          field,
          (field in aspirations ? aspirations : profile)[
            field as keyof typeof aspirations | keyof typeof profile
          ] as string,
        ),
      ]),
    ) as Errors
    if (Object.values(validationErrors).some((error) => error)) {
      setErrors(validationErrors)
      return
    }

    setLoading(true)
    const formData = new FormData()
    formData.append("aspirations", JSON.stringify(aspirations))
    formData.append("profile", JSON.stringify(profile))
    if (profile.resume) formData.append("resume", profile.resume)

    try {
      const response = await fetch("http://localhost:8000/api/profile/", {
        method: "POST",
        body: formData,
      })
      const data = await response.json()
      if (data.error) throw new Error(data.details || data.error)

      setResult({
        gapScore: data.gap_score,
        gapDetails: data.gap_details,
        microOkrs: data.micro_okrs,
        report: data.report,
      })
    } catch (error) {
      console.error("Error submitting form:", error)
      setResult({
        gapScore: 0,
        gapDetails: {},
        microOkrs: [],
        report: { summary: "Error generating report", feedback: "", next_steps: "" },
      })
    } finally {
      setLoading(false)
    }
  }

  // Helper function to safely render any value
  const renderValue = (value: any): string => {
    if (value === null || value === undefined) {
      return "N/A"
    }
    if (typeof value === "object") {
      return JSON.stringify(value, null, 2)
    }
    return String(value)
  }

  // Get icon for gap analysis categories
  const getCategoryIcon = (category: string) => {
    const normalizedCategory = category.toLowerCase()
    switch (normalizedCategory) {
      case "ctc":
      case "salary":
        return <DollarSign className="h-7 w-7 text-green-600" />
      case "skills":
      case "technical":
        return <Code className="h-7 w-7 text-blue-600" />
      case "experience":
      case "work":
        return <Briefcase className="h-7 w-7 text-purple-600" />
      case "education":
      case "degree":
        return <Award className="h-7 w-7 text-indigo-600" />
      default:
        return <BarChart3 className="h-7 w-7 text-gray-600" />
    }
  }

  // Render enhanced result
  const renderEnhancedResult = () => {
    if (!result) return null

    return (
      <div className="mt-12 space-y-10 animate-fade-in">
        {/* Gap Score Card */}
        <div className="bg-gradient-to-r from-yellow-400 via-yellow-500 to-amber-500 p-8 lg:p-12 rounded-3xl shadow-2xl relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-white bg-opacity-10 rounded-full -mr-16 -mt-16"></div>
          <div className="absolute bottom-0 left-0 w-24 h-24 bg-black bg-opacity-10 rounded-full -ml-12 -mb-12"></div>
          <div className="relative z-10">
            <div className="flex flex-col lg:flex-row items-center justify-between space-y-6 lg:space-y-0">
              <div className="flex items-center space-x-6 text-center lg:text-left">
                <div className="p-4 bg-black bg-opacity-20 rounded-2xl">
                  <TrendingUp className="h-12 w-12 text-black" />
                </div>
                <div>
                  <h3 className="text-2xl lg:text-3xl font-bold text-black mb-2">Overall Gap Score</h3>
                  <p className="text-black opacity-80 text-lg">Your market readiness assessment</p>
                </div>
              </div>
              <div className="text-center">
                <p className="text-5xl lg:text-6xl font-bold text-black mb-2">{result.gapScore}</p>
                <p className="text-xl text-black opacity-80">out of 100</p>
              </div>
            </div>
            <div className="mt-8 bg-black bg-opacity-20 rounded-full h-6 overflow-hidden">
              <div
                className="bg-black h-6 rounded-full transition-all duration-2000 ease-out shadow-lg"
                style={{ width: `${Math.min(Math.max(result.gapScore, 0), 100)}%` }}
              ></div>
            </div>
          </div>
        </div>

        {/* Enhanced Gap Analysis */}
        {result.gapDetails && Object.keys(result.gapDetails).length > 0 && (
          <div className="bg-white p-8 lg:p-12 rounded-3xl shadow-2xl border border-gray-100">
            <div className="flex items-center space-x-4 mb-10">
              <div className="p-3 bg-yellow-100 rounded-2xl">
                <BarChart3 className="h-10 w-10 text-yellow-600" />
              </div>
              <div>
                <h3 className="text-2xl lg:text-3xl font-bold text-black mb-2">Detailed Gap Analysis</h3>
                <p className="text-gray-600 text-lg">Comprehensive breakdown of your profile alignment</p>
              </div>
            </div>

            <div className="grid gap-8 lg:gap-10 xl:grid-cols-2">
              {Object.entries(result.gapDetails).map(([category, data]) => (
                <GapAnalysisItem
                  key={category}
                  title={category.charAt(0).toUpperCase() + category.slice(1)}
                  data={data}
                  icon={getCategoryIcon(category)}
                />
              ))}
            </div>
          </div>
        )}

        {/* Enhanced Micro OKRs */}
        {result.microOkrs && result.microOkrs.length > 0 && (
          <div className="bg-white p-8 lg:p-12 rounded-3xl shadow-2xl border border-gray-100">
            <div className="flex items-center space-x-4 mb-10">
              <div className="p-3 bg-yellow-100 rounded-2xl">
                <Target className="h-10 w-10 text-yellow-600" />
              </div>
              <div>
                <h3 className="text-2xl lg:text-3xl font-bold text-black mb-2">Action Plan</h3>
                <p className="text-gray-600 text-lg">Personalized micro-OKRs to bridge your gaps</p>
              </div>
            </div>
            <div className="grid gap-8">
              {result.microOkrs.map((okr, index) => (
                <div
                  key={index}
                  className="bg-gradient-to-r from-yellow-50 via-amber-50 to-yellow-100 p-8 rounded-2xl border-l-4 border-yellow-400 hover:shadow-lg transition-all duration-300 animate-fade-in-up"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <div className="flex items-start space-x-6">
                    <div className="bg-yellow-400 text-black rounded-full w-12 h-12 flex items-center justify-center font-bold text-lg shadow-lg">
                      {index + 1}
                    </div>
                    <div className="flex-1">
                      <h4 className="text-xl lg:text-2xl font-bold text-black mb-4">{renderValue(okr.task)}</h4>
                      <div className="grid md:grid-cols-2 gap-6">
                        <div className="flex items-center space-x-3 text-gray-700">
                          <div className="p-2 bg-yellow-200 rounded-lg">
                            <BookOpen className="h-5 w-5 text-yellow-700" />
                          </div>
                          <div>
                            <span className="font-semibold text-black">Resource:</span>
                            <p className="text-gray-600">{renderValue(okr.resource)}</p>
                          </div>
                        </div>
                        <div className="flex items-center space-x-3 text-gray-700">
                          <div className="p-2 bg-yellow-200 rounded-lg">
                            <Calendar className="h-5 w-5 text-yellow-700" />
                          </div>
                          <div>
                            <span className="font-semibold text-black">Timeline:</span>
                            <p className="text-gray-600">{renderValue(okr.timeline)}</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Enhanced Coaching Report */}
        <div className="bg-gradient-to-br from-black via-gray-900 to-black p-8 lg:p-12 rounded-3xl shadow-2xl text-white relative overflow-hidden">
          <div className="absolute top-0 right-0 w-40 h-40 bg-yellow-400 bg-opacity-10 rounded-full -mr-20 -mt-20"></div>
          <div className="absolute bottom-0 left-0 w-32 h-32 bg-yellow-400 bg-opacity-5 rounded-full -ml-16 -mb-16"></div>
          <div className="relative z-10">
            <div className="flex items-center space-x-4 mb-10">
              <div className="p-3 bg-yellow-400 rounded-2xl">
                <Sparkles className="h-10 w-10 text-black" />
              </div>
              <div>
                <h3 className="text-2xl lg:text-3xl font-bold mb-2">AI Coaching Report</h3>
                <p className="text-gray-300 text-lg">Personalized insights and recommendations</p>
              </div>
            </div>
            <div className="grid gap-8">
              <div className="bg-gray-800 p-8 rounded-2xl border border-gray-700 hover:border-yellow-400 transition-all duration-300">
                <h4 className="text-xl lg:text-2xl font-bold text-yellow-400 mb-6 flex items-center space-x-3">
                  <BarChart3 className="h-6 w-6" />
                  <span>Summary</span>
                </h4>
                <p className="text-gray-300 leading-relaxed text-lg">{renderValue(result.report.summary)}</p>
              </div>
              <div className="bg-gray-800 p-8 rounded-2xl border border-gray-700 hover:border-yellow-400 transition-all duration-300">
                <h4 className="text-xl lg:text-2xl font-bold text-yellow-400 mb-6 flex items-center space-x-3">
                  <AlertTriangle className="h-6 w-6" />
                  <span>Feedback</span>
                </h4>
                <p className="text-gray-300 leading-relaxed text-lg">{renderValue(result.report.feedback)}</p>
              </div>
              <div className="bg-gray-800 p-8 rounded-2xl border border-gray-700 hover:border-yellow-400 transition-all duration-300">
                <h4 className="text-xl lg:text-2xl font-bold text-yellow-400 mb-6 flex items-center space-x-3">
                  <CheckCircle className="h-6 w-6" />
                  <span>Next Steps</span>
                </h4>
                <p className="text-gray-300 leading-relaxed text-lg">{renderValue(result.report.next_steps)}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <>
      <Navbar />
      <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-100">
        {/* Enhanced Hero Section */}
        <div className="bg-gradient-to-r from-black via-gray-900 to-black text-white py-20 lg:py-32 relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-yellow-400/10 to-transparent"></div>
          <div className="absolute top-0 right-0 w-96 h-96 bg-yellow-400 bg-opacity-5 rounded-full -mr-48 -mt-48"></div>
          <div className="absolute bottom-0 left-0 w-64 h-64 bg-yellow-400 bg-opacity-5 rounded-full -ml-32 -mb-32"></div>
          <div className="max-w-7xl mx-auto px-6 text-center relative z-10">
            <div className="flex justify-center mb-8">
              <div className="p-4 bg-yellow-400 rounded-2xl animate-pulse">
                <Target className="h-16 w-16 text-black" />
              </div>
            </div>
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-6 animate-fade-in">
              Reality-Matching <span className="text-yellow-400">Simulator</span>
            </h1>
            <p className="text-xl lg:text-2xl text-gray-300 mb-12 max-w-4xl mx-auto leading-relaxed animate-fade-in-up">
              AI-powered platform to align your placement expectations with market realities through intelligent
              analysis and personalized guidance
            </p>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6 max-w-4xl mx-auto">
              {[
                { name: "AI Analysis", icon: <Brain className="h-5 w-5" />, color: "from-blue-500 to-blue-600" },
                {
                  name: "Gap Assessment",
                  icon: <BarChart3 className="h-5 w-5" />,
                  color: "from-green-500 to-green-600",
                },
                { name: "Action Plans", icon: <Target className="h-5 w-5" />, color: "from-purple-500 to-purple-600" },
                {
                  name: "Career Coaching",
                  icon: <Award className="h-5 w-5" />,
                  color: "from-orange-500 to-orange-600",
                },
              ].map((feature, index) => (
                <div
                  key={feature.name}
                  className="group relative bg-white bg-opacity-10 backdrop-blur-sm border border-white border-opacity-20 px-6 py-4 rounded-2xl hover:bg-opacity-20 transition-all duration-300 transform hover:scale-105 animate-fade-in-up"
                  style={{ animationDelay: `${index * 150}ms` }}
                >
                  <div className="flex flex-col items-center space-y-2 text-center">
                    <div
                      className={`p-2 rounded-xl bg-gradient-to-r ${feature.color} shadow-lg group-hover:shadow-xl transition-shadow duration-300`}
                    >
                      {feature.icon}
                    </div>
                    <span className="text-sm lg:text-base font-semibold text-white group-hover:text-yellow-300 transition-colors duration-300">
                      {feature.name}
                    </span>
                  </div>

                  {/* Subtle glow effect */}
                  <div
                    className={`absolute inset-0 rounded-2xl bg-gradient-to-r ${feature.color} opacity-0 group-hover:opacity-20 transition-opacity duration-300 -z-10`}
                  ></div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Enhanced Main Form */}
        <div className="max-w-7xl mx-auto px-6 py-16 lg:py-24">
          <div className="bg-white rounded-3xl shadow-2xl p-8 lg:p-16 border border-gray-100 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-yellow-400 bg-opacity-5 rounded-full -mr-16 -mt-16"></div>
            <div className="text-center mb-12 relative z-10">
              <h2 className="text-3xl lg:text-4xl font-bold text-black mb-4">Enter Your Profile</h2>
              <p className="text-gray-600 text-lg lg:text-xl max-w-2xl mx-auto">
                Let's analyze your career aspirations and current profile to provide personalized insights
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-12" encType="multipart/form-data">
              {/* Enhanced Aspirations Section */}
              <div className="bg-gradient-to-br from-yellow-50 to-amber-50 p-8 lg:p-10 rounded-3xl border-2 border-yellow-200 relative overflow-hidden">
                <div className="absolute top-0 right-0 w-24 h-24 bg-yellow-400 bg-opacity-10 rounded-full -mr-12 -mt-12"></div>
                <div className="relative z-10">
                  <h3 className="text-2xl lg:text-3xl font-bold text-black mb-8 flex items-center space-x-4">
                    <div className="p-3 bg-yellow-400 rounded-2xl">
                      <Target className="h-8 w-8 text-black" />
                    </div>
                    <span>Career Aspirations</span>
                  </h3>

                  <div className="grid lg:grid-cols-2 gap-8 mb-8">
                    <InputField
                      type="text"
                      placeholder="e.g., 12 LPA, 15 LPA, 20 LPA"
                      value={aspirations.ctc}
                      onChange={(e) => handleInputChange("ctc", e.target.value)}
                      error={errors.ctc}
                      label="Desired CTC"
                      icon={<DollarSign className="h-5 w-5 text-yellow-600" />}
                    />
                    <InputField
                      type="text"
                      placeholder="e.g., Software Engineer, Data Scientist"
                      value={aspirations.role}
                      onChange={(e) => handleInputChange("role", e.target.value)}
                      error={errors.role}
                      label="Desired Role"
                      icon={<Briefcase className="h-5 w-5 text-yellow-600" />}
                    />
                  </div>

                  <div className="space-y-8">
                    <InputField
                      type="text"
                      placeholder="e.g., Google, Microsoft, Amazon, startups"
                      value={aspirations.companies}
                      onChange={(e) => handleInputChange("companies", e.target.value)}
                      error={errors.companies}
                      label="Target Companies"
                      icon={<Building className="h-5 w-5 text-yellow-600" />}
                    />
                    <InputField
                      type="text"
                      placeholder="e.g., Complete DSA course, Build 3 projects, Learn system design"
                      value={aspirations.okrs}
                      onChange={(e) => handleInputChange("okrs", e.target.value)}
                      error={errors.okrs}
                      label="Personal OKRs"
                      icon={<Target className="h-5 w-5 text-yellow-600" />}
                    />
                  </div>
                </div>
              </div>

              {/* Enhanced Profile Section */}
              <div className="bg-gradient-to-br from-gray-50 to-slate-50 p-8 lg:p-10 rounded-3xl border-2 border-gray-200 relative overflow-hidden">
                <div className="absolute top-0 right-0 w-24 h-24 bg-gray-400 bg-opacity-10 rounded-full -mr-12 -mt-12"></div>
                <div className="relative z-10">
                  <h3 className="text-2xl lg:text-3xl font-bold text-black mb-8 flex items-center space-x-4">
                    <div className="p-3 bg-gray-600 rounded-2xl">
                      <Upload className="h-8 w-8 text-white" />
                    </div>
                    <span>Your Profile</span>
                  </h3>

                  <div className="mb-8">
                    <FileInputField
                      onChange={(e) => handleInputChange("resume", e.target.files ? e.target.files[0] : null)}
                      error={errors.profile}
                      label="Upload Resume"
                      accept=".pdf,.doc,.docx"
                      icon={<Upload className="h-5 w-5 text-gray-600" />}
                    />
                  </div>

                  <div className="grid md:grid-cols-3 gap-8">
                    <InputField
                      type="text"
                      placeholder="https://linkedin.com/in/yourprofile"
                      value={profile.linkedin}
                      onChange={(e) => handleInputChange("linkedin", e.target.value)}
                      error={errors.linkedin}
                      label="LinkedIn"
                      icon={<Linkedin className="h-5 w-5 text-blue-600" />}
                    />
                    <InputField
                      type="text"
                      placeholder="https://github.com/yourusername"
                      value={profile.github}
                      onChange={(e) => handleInputChange("github", e.target.value)}
                      error={errors.github}
                      label="GitHub"
                      icon={<Github className="h-5 w-5 text-gray-800" />}
                    />
                    <InputField
                      type="text"
                      placeholder="https://leetcode.com/yourusername"
                      value={profile.leetcode}
                      onChange={(e) => handleInputChange("leetcode", e.target.value)}
                      error={errors.leetcode}
                      label="LeetCode"
                      icon={<Code className="h-5 w-5 text-orange-600" />}
                    />
                  </div>
                </div>
              </div>

              {/* Enhanced Submit Button */}
              <div className="text-center">
                <button
                  type="submit"
                  className="w-full lg:w-auto lg:px-16 bg-gradient-to-r from-yellow-400 via-yellow-500 to-amber-500 text-black p-6 lg:p-8 rounded-2xl font-bold text-xl lg:text-2xl hover:from-yellow-500 hover:to-amber-600 transition-all duration-300 transform hover:scale-105 shadow-2xl disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none relative overflow-hidden group"
                  disabled={loading}
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-0 group-hover:opacity-20 transform -skew-x-12 group-hover:animate-shimmer"></div>
                  {loading ? (
                    <span className="flex items-center justify-center space-x-3">
                      <Loader2 className="animate-spin h-8 w-8" />
                      <span>Analyzing Your Profile...</span>
                    </span>
                  ) : (
                    <span className="flex items-center justify-center space-x-3">
                      <Zap className="h-8 w-8" />
                      <span>Analyze My Profile</span>
                    </span>
                  )}
                </button>
              </div>
            </form>

            {/* Results */}
            {renderEnhancedResult()}
          </div>
        </div>
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

        @keyframes shimmer {
          0% {
            transform: translateX(-100%) skewX(-12deg);
          }
          100% {
            transform: translateX(200%) skewX(-12deg);
          }
        }

        .animate-fade-in {
          animation: fade-in 0.6s ease-out;
        }

        .animate-fade-in-up {
          animation: fade-in-up 0.8s ease-out;
        }

        .animate-shimmer {
          animation: shimmer 1.5s ease-in-out;
        }
      `}</style>
    </>
  )
}

export default Home
