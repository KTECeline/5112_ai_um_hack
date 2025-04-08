"use client"

import { useEffect, useState } from "react"
import { Area, AreaChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

// Generate sample DeFi yield curve data
const generateEquityCurveData = () => {
  const data = []
  const now = new Date()
  let defiYield = 10000 // Starting DeFi portfolio value
  let ethHodl = 10000 // Starting ETH HODL value

  for (let i = 30; i >= 0; i--) {
    const date = new Date(now)
    date.setDate(now.getDate() - i)

    // Create a growing DeFi yield curve with some volatility
    const defiDailyReturn = Math.random() * 0.04 - 0.01 // Between -1% and 3%
    defiYield = defiYield * (1 + defiDailyReturn)

    // ETH HODL grows more slowly with higher volatility
    const ethDailyReturn = Math.random() * 0.05 - 0.02 // Between -2% and 3%
    ethHodl = ethHodl * (1 + ethDailyReturn)

    data.push({
      date: date.toISOString().split("T")[0],
      defiYield: defiYield.toFixed(2),
      ethHodl: ethHodl.toFixed(2),
    })
  }

  return data
}

export function EquityCurveChart() {
  const [data, setData] = useState([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Simulate data loading
    const timer = setTimeout(() => {
      setData(generateEquityCurveData())
      setIsLoading(false)
    }, 500)

    return () => clearTimeout(timer)
  }, [])

  if (isLoading) {
    return (
      <div className="flex h-80 items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-gray-700 border-t-blue-500"></div>
      </div>
    )
  }

  return (
    <div className="h-80 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="colorDefiYield" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="colorEthHodl" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="date"
            tick={{ fill: "#9ca3af" }}
            tickLine={{ stroke: "#4b5563" }}
            axisLine={{ stroke: "#4b5563" }}
            tickFormatter={(value) => {
              const date = new Date(value)
              return `${date.getDate()}/${date.getMonth() + 1}`
            }}
          />
          <YAxis
            tick={{ fill: "#9ca3af" }}
            tickLine={{ stroke: "#4b5563" }}
            axisLine={{ stroke: "#4b5563" }}
            domain={["auto", "auto"]}
            tickFormatter={(value) => `$${(value / 1000).toFixed(1)}k`}
          />
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1f2937",
              borderColor: "#4b5563",
              color: "#e5e7eb",
              borderRadius: "0.375rem",
            }}
            labelStyle={{ color: "#e5e7eb" }}
            formatter={(value, name) => [
              `${Number(value).toLocaleString()} USD`,
              name === "defiYield" ? "AlphaPulse DeFi Strategy" : "ETH HODL",
            ]}
            labelFormatter={(value) => `Block Date: ${new Date(value).toLocaleDateString()}`}
          />
          <Legend
            wrapperStyle={{ color: "#e5e7eb" }}
            formatter={(value) => (value === "defiYield" ? "AlphaPulse DeFi Strategy" : "ETH HODL")}
          />
          <Area
            type="monotone"
            dataKey="defiYield"
            stroke="#8b5cf6"
            fillOpacity={1}
            fill="url(#colorDefiYield)"
            strokeWidth={2}
            animationDuration={1000}
          />
          <Area
            type="monotone"
            dataKey="ethHodl"
            stroke="#3b82f6"
            fillOpacity={1}
            fill="url(#colorEthHodl)"
            strokeWidth={2}
            animationDuration={1000}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
