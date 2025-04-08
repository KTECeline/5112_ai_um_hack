"use client"

import { useEffect, useState } from "react"
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

// Update the chart title and data generation to be more Web3 focused

// Generate some sample data
const generateData = () => {
  const data = []
  const now = new Date()

  for (let i = 30; i >= 0; i--) {
    const date = new Date(now)
    date.setDate(now.getDate() - i)

    // Create a sine wave with some noise for the price
    const baseValue = 3000 // ETH price in USD
    const amplitude = 300
    const period = 30
    const noise = Math.random() * 100 - 50

    const price = baseValue + amplitude * Math.sin((2 * Math.PI * i) / period) + noise

    // Add some volume data and gas data
    const volume = Math.floor(Math.random() * 1000) + 500
    const gasPrice = Math.floor(Math.random() * 100) + 20 // in gwei

    data.push({
      date: date.toISOString().split("T")[0],
      price: price.toFixed(2),
      volume: volume,
      gasPrice: gasPrice,
    })
  }

  return data
}

export function MarketTrendChart() {
  const [data, setData] = useState([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Simulate data loading
    const timer = setTimeout(() => {
      setData(generateData())
      setIsLoading(false)
    }, 500)

    return () => clearTimeout(timer)
  }, [])

  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-gray-700 border-t-blue-500"></div>
      </div>
    )
  }

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
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
            tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
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
            itemStyle={{ color: "#a78bfa" }}
            formatter={(value, name) => {
              if (name === "price") return [`${Number(value).toLocaleString()} USD`, "ETH Price"]
              if (name === "gasPrice") return [`${value} gwei`, "Gas Price"]
              return [value, name]
            }}
            labelFormatter={(value) => `Block Date: ${new Date(value).toLocaleDateString()}`}
          />
          <Area
            type="monotone"
            dataKey="price"
            stroke="#8b5cf6"
            fillOpacity={1}
            fill="url(#colorPrice)"
            strokeWidth={2}
            animationDuration={1000}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
