"use client"

import { useEffect, useState } from "react"
import { ArrowUpRight, ArrowDownRight } from "lucide-react"

interface TokenPrice {
  symbol: string
  name: string
  price: number
  change: number
}

export function TokenPriceTicker() {
  const [prices, setPrices] = useState<TokenPrice[]>([
    { symbol: "ETH", name: "Ethereum", price: 3245.67, change: 2.34 },
    { symbol: "BTC", name: "Bitcoin", price: 52345.89, change: -1.23 },
    { symbol: "SOL", name: "Solana", price: 123.45, change: 5.67 },
    { symbol: "AVAX", name: "Avalanche", price: 34.56, change: 3.21 },
    { symbol: "MATIC", name: "Polygon", price: 1.23, change: -0.45 },
  ])

  useEffect(() => {
    // Simulate price updates
    const interval = setInterval(() => {
      setPrices((currentPrices) =>
        currentPrices.map((token) => {
          const randomChange = (Math.random() * 2 - 1) * 0.5
          const newChange = Number.parseFloat((token.change + randomChange).toFixed(2))
          const changePercent = 1 + newChange / 100
          const newPrice = Number.parseFloat((token.price * changePercent).toFixed(2))
          return { ...token, price: newPrice, change: newChange }
        }),
      )
    }, 5000)

    return () => clearInterval(interval)
  }, [])
  return (
    <div className="w-full overflow-hidden bg-gray-900/50 border-y border-gray-800 backdrop-blur-sm">
      <div className="flex animate-marquee whitespace-nowrap py-2">
        {/* Duplicate the prices array to create a continuous loop */}
        {[...prices, ...prices].map((token, index) => (
          <div key={index} className="mx-4 flex items-center">
            <div className="flex h-6 w-6 items-center justify-center rounded-full bg-gray-800 text-xs font-bold">
              {token.symbol.charAt(0)}
            </div>
            <span className="ml-2 font-medium text-white">{token.symbol}</span>
            <span className="ml-2 text-gray-400">${token.price.toLocaleString()}</span>
            <div className={`ml-2 flex items-center ${token.change >= 0 ? "text-green-400" : "text-red-400"}`}>
              {token.change >= 0 ? <ArrowUpRight className="h-3 w-3" /> : <ArrowDownRight className="h-3 w-3" />}
              <span className="ml-1 text-xs">{Math.abs(token.change)}%</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}