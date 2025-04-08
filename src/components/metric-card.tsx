import type React from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowDownIcon, ArrowUpIcon } from "lucide-react"

interface MetricCardProps {
  title: string
  value: string
  change: string
  trend: "up" | "down" | "neutral"
  description: string
  icon: React.ReactNode
  color: "blue" | "purple" | "pink"
}

export function MetricCard({ title, value, change, trend, description, icon, color }: MetricCardProps) {
  const getGradient = () => {
    switch (color) {
      case "blue":
        return "from-blue-600 to-blue-400"
      case "purple":
        return "from-purple-600 to-purple-400"
      case "pink":
        return "from-pink-600 to-pink-400"
      default:
        return "from-blue-600 to-blue-400"
    }
  }

  const getTextColor = () => {
    switch (color) {
      case "blue":
        return "text-blue-400"
      case "purple":
        return "text-purple-400"
      case "pink":
        return "text-pink-400"
      default:
        return "text-blue-400"
    }
  }

  const getBgColor = () => {
    switch (color) {
      case "blue":
        return "bg-blue-500/10"
      case "purple":
        return "bg-purple-500/10"
      case "pink":
        return "bg-pink-500/10"
      default:
        return "bg-blue-500/10"
    }
  }

  return (
    <Card className="border-gray-800 bg-gray-900/50 backdrop-blur-sm">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium text-gray-400">{title}</CardTitle>
          <div className={`rounded-full p-1 ${getBgColor()}`}>{icon}</div>
        </div>
        <CardDescription className="text-xs text-gray-500">{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-baseline justify-between">
          <div className={`text-2xl font-bold ${getTextColor()}`}>{value}</div>
          <div className="flex items-center gap-1 text-sm">
            {trend === "up" ? (
              <div className="flex items-center text-green-400">
                <ArrowUpIcon className="h-3 w-3" />
                {change}
              </div>
            ) : (
              <div className="flex items-center text-red-400">
                <ArrowDownIcon className="h-3 w-3" />
                {change}
              </div>
            )}
          </div>
        </div>
        <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-gray-800">
          <div className={`h-full rounded-full bg-gradient-to-r ${getGradient()}`} style={{ width: "70%" }}></div>
        </div>
      </CardContent>
    </Card>
  )
}
