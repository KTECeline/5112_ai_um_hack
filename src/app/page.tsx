import Link from "next/link"
import { ArrowRight, BarChart3, LineChart, TrendingUp } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { MarketTrendChart } from "@/components/market-trend-chart"
import { EquityCurveChart } from "@/components/equity-curve-chart"
import { ParticleBackground } from "@/components/particle-background"
import { MetricCard } from "@/components/metric-card"
import { RegimeIndicator } from "@/components/regime-indicator"
import { WalletConnectButton } from "@/components/wallet-connect-button"
import { TokenPriceTicker } from "@/components/token-price-ticker"


export default function Home() {
  return (
    <div className="relative min-h-screen w-full overflow-hidden bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      <ParticleBackground />

      <header className="relative z-10 border-b border-gray-800 bg-black/20 backdrop-blur-md">
        <div className="mx-auto flex h-24 items-center justify-between px-6 sm:px-8 lg:px-10 max-w-[1800px]">
          <Link href="/" className="flex items-center gap-4">
            <div className="relative h-12 w-12 overflow-hidden rounded-full bg-gradient-to-r from-pink-500 via-purple-500 to-blue-500">
              <div className="absolute inset-0.5 rounded-full bg-gray-950"></div>
            </div>
            <span className="text-3xl font-bold tracking-tight text-white">AlphaPulse</span>
          </Link>
          <nav className="hidden gap-10 md:flex">
            <Link href="#" className="text-lg font-medium text-gray-400 transition-colors hover:text-white">
              Features
            </Link>
            <Link href="#" className="text-lg font-medium text-gray-400 transition-colors hover:text-white">
              Strategies
            </Link>
            <Link href="#" className="text-lg font-medium text-gray-400 transition-colors hover:text-white">
              Pricing
            </Link>
            <Link href="#" className="text-lg font-medium text-gray-400 transition-colors hover:text-white">
              Documentation
            </Link>
          </nav>
          <div className="flex items-center gap-8">
            <Button variant="ghost" className="hidden text-lg text-gray-400 hover:text-white md:flex">
              Sign In
            </Button>
            <Button className="bg-gradient-to-r from-blue-600 to-purple-600 text-xl text-white hover:from-blue-700 hover:to-purple-700">
              Get Started
            </Button>
          </div>
        </div>
      </header>

      <TokenPriceTicker />

      <main className="relative z-10">
        <section className="mx-auto px-6 py-20 sm:px-8 lg:px-10 max-w-[1800px]">
          <div className="grid gap-16 md:grid-cols-2 md:gap-20">
            <div className="flex flex-col justify-center space-y-6">
              <div className="inline-flex items-center rounded-full border border-purple-800/30 bg-purple-500/10 px-4 py-2 text-lg text-purple-300">
                <span className="mr-3 h-3 w-3 rounded-full bg-purple-400"></span>
                Next-gen AI Trading Platform
              </div>
              <h1 className="text-6xl font-bold tracking-tighter text-white sm:text-7xl md:text-8xl">
                Trade Smarter with{" "}
                <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                  AI-Powered
                </span>{" "}
                Insights
              </h1>
              <p className="max-w-[700px] text-2xl text-gray-400 md:text-3xl">
                AlphaPulse leverages advanced machine learning to identify market regimes and execute optimal trading
                strategies in real-time.
              </p>
              <div className="flex flex-col gap-6 sm:flex-row">
                <Button className="group bg-gradient-to-r from-blue-600 to-purple-600 text-xl text-white hover:from-blue-700 hover:to-purple-700">
                  View Strategy Demo
                  <ArrowRight className="ml-3 h-6 w-6 transition-transform group-hover:translate-x-1" />
                </Button>
                <Button variant="outline" className="border-gray-700 text-xl text-gray-300 hover:bg-gray-800 hover:text-white">
                  Run Prediction
                </Button>
              </div>
            </div>
            <div className="relative flex items-center justify-center">
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="h-96 w-96 rounded-full bg-purple-500/20 blur-3xl"></div>
                <div className="absolute h-80 w-80 rounded-full bg-blue-500/20 blur-3xl"></div>
              </div>
              <Card className="relative w-full overflow-hidden border-gray-800 bg-gray-900/50 backdrop-blur-sm">
                <CardHeader className="border-b border-gray-800 bg-black/30 px-8 py-6">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-3xl font-medium text-white">Market Trend Analysis</CardTitle>
                    <RegimeIndicator regime="bullish" />
                  </div>
                </CardHeader>
                <CardContent className="p-8">
                  <MarketTrendChart />
                </CardContent>
              </Card>
            </div>
          </div>
        </section>

        <section className="mx-auto px-6 py-20 sm:px-8 lg:px-10 max-w-[1800px]">
          <div className="mb-12 flex flex-col items-center justify-center text-center">
            <h2 className="text-5xl font-bold tracking-tight text-white sm:text-6xl">Real-Time Trading Metrics</h2>
            <p className="mt-6 max-w-[900px] text-2xl text-gray-400">
              Monitor your portfolio performance with advanced analytics and AI-driven insights
            </p>
          </div>

          <div className="grid gap-10 sm:grid-cols-2 lg:grid-cols-3">
            <MetricCard
              title="Sharpe Ratio"
              value="2.87"
              change="+0.32"
              trend="up"
              description="Risk-adjusted return"
              icon={<BarChart3 className="h-6 w-6" />}
              color="blue"
            />
            <MetricCard
              title="Max Drawdown"
              value="12.4%"
              change="-3.1%"
              trend="up"
              description="Peak-to-trough decline"
              icon={<TrendingUp className="h-6 w-6" />}
              color="pink"
            />
            <MetricCard
              title="Win Rate"
              value="68.5%"
              change="+2.3%"
              trend="up"
              description="Successful trades ratio"
              icon={<LineChart className="h-6 w-6" />}
              color="purple"
            />
          </div>

          <div className="mt-12">
            <Card className="border-gray-800 bg-gray-900/50 backdrop-blur-sm">
              <CardHeader className="border-b border-gray-800 bg-black/30">
                <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                  <CardTitle className="text-3xl font-medium text-white">Portfolio Performance</CardTitle>
                  <Tabs defaultValue="1m" className="w-[500px]">
                    <TabsList className="bg-gray-800/50">
                      <TabsTrigger value="1w" className="data-[state=active]:bg-gray-700">
                        1W
                      </TabsTrigger>
                      <TabsTrigger value="1m" className="data-[state=active]:bg-gray-700">
                        1M
                      </TabsTrigger>
                      <TabsTrigger value="3m" className="data-[state=active]:bg-gray-700">
                        3M
                      </TabsTrigger>
                      <TabsTrigger value="1y" className="data-[state=active]:bg-gray-700">
                        1Y
                      </TabsTrigger>
                      <TabsTrigger value="all" className="data-[state=active]:bg-gray-700">
                        All
                      </TabsTrigger>
                    </TabsList>
                  </Tabs>
                </div>
                <CardDescription className="text-xl text-gray-400">
                  Equity curve and cumulative returns over time
                </CardDescription>
              </CardHeader>
              <CardContent className="p-8">
                <EquityCurveChart />
              </CardContent>
            </Card>
          </div>
        </section>

        <section className="mx-auto px-4 py-16 sm:px-6 lg:px-8 max-w-7xl">
          <div className="grid gap-8 md:grid-cols-2 md:gap-12">
            <div className="order-2 md:order-1">
              <Card className="border-gray-800 bg-gray-900/50 backdrop-blur-sm">
                <CardHeader className="border-b border-gray-800 bg-black/30">
                  <CardTitle className="text-xl font-medium text-white">AI Strategy Features</CardTitle>
                </CardHeader>
                <CardContent className="p-6">
                  <ul className="grid gap-4">
                    {[
                      "Cross-chain liquidity optimization",
                      "MEV-resistant transaction routing",
                      "Smart contract interaction automation",
                      "On-chain sentiment analysis",
                      "Gas-optimized position management",
                      "Decentralized oracle integration",
                    ].map((feature, index) => (
                      <li key={index} className="flex items-center gap-3">
                        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-r from-blue-600 to-purple-600">
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="24"
                            height="24"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            className="h-4 w-4 text-white"
                          >
                            <polyline points="20 6 9 17 4 12"></polyline>
                          </svg>
                        </div>
                        <span className="text-gray-300">{feature}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </div>
            <div className="order-1 flex flex-col justify-center space-y-4 md:order-2">
              <div className="inline-flex items-center rounded-full border border-blue-800/30 bg-blue-500/10 px-3 py-1 text-sm text-blue-300">
                <span className="mr-2 h-2 w-2 rounded-full bg-blue-400"></span>
                Web3-Native AI Technology
              </div>
              <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
                Harness the Power of On-Chain Analytics
              </h2>
              <p className="max-w-[600px] text-gray-400">
                Our proprietary algorithms analyze thousands of on-chain data points to identify market regimes and
                predict price movements with unprecedented accuracy across decentralized exchanges.
              </p>
              <ul className="grid gap-3">
                {[
                  "Real-time blockchain data analysis",
                  "Smart contract interaction optimization",
                  "Cross-chain liquidity aggregation",
                ].map((item, index) => (
                  <li key={index} className="flex items-center gap-2 text-gray-300">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="h-5 w-5 text-blue-400"
                    >
                      <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    {item}
                  </li>
                ))}
              </ul>
              <div>
                <Button className="bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700">
                  Explore Web3 Features
                </Button>
              </div>
            </div>
          </div>
        </section>

        <section className="relative">
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="h-96 w-96 rounded-full bg-purple-500/10 blur-3xl"></div>
            <div className="absolute h-64 w-64 rounded-full bg-blue-500/10 blur-3xl"></div>
          </div>
          <div className="relative mx-auto px-4 py-16 sm:px-6 lg:px-8 max-w-7xl">
            <div className="mx-auto max-w-3xl rounded-2xl border border-gray-800 bg-gray-900/50 p-8 text-center backdrop-blur-md md:p-12">
              <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
                Ready to Enter the Future of DeFi?
              </h2>
              <p className="mx-auto mt-4 max-w-[600px] text-gray-400">
                Join thousands of Web3 traders who are already leveraging AI to gain an edge in decentralized markets.
              </p>
              <div className="mt-8 flex flex-col gap-4 sm:flex-row sm:justify-center">
                <Button className="bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700">
                  Connect Wallet
                </Button>
                <Button variant="outline" className="border-gray-700 text-gray-300 hover:bg-gray-800 hover:text-white">
                  Try Demo dApp
                </Button>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="relative z-10 border-t border-gray-800 bg-black/20 backdrop-blur-md">
        <div className="mx-auto px-4 py-8 sm:px-6 lg:px-8 max-w-7xl">
          <div className="grid gap-8 sm:grid-cols-2 md:grid-cols-4">
            <div>
              <h3 className="mb-4 text-lg font-medium text-white">AlphaPulse</h3>
              <ul className="grid gap-2">
                <li>
                  <Link href="#" className="text-sm text-gray-400 transition-colors hover:text-white">
                    About Us
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-sm text-gray-400 transition-colors hover:text-white">
                    Governance
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-sm text-gray-400 transition-colors hover:text-white">
                    Tokenomics
                  </Link>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="mb-4 text-lg font-medium text-white">Products</h3>
              <ul className="grid gap-2">
                <li>
                  <Link href="#" className="text-sm text-gray-400 transition-colors hover:text-white">
                    DEX Aggregator
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-sm text-gray-400 transition-colors hover:text-white">
                    Yield Optimizer
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-sm text-gray-400 transition-colors hover:text-white">
                    NFT Analytics
                  </Link>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="mb-4 text-lg font-medium text-white">Resources</h3>
              <ul className="grid gap-2">
                <li>
                  <Link href="#" className="text-sm text-gray-400 transition-colors hover:text-white">
                    Documentation
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-sm text-gray-400 transition-colors hover:text-white">
                    Smart Contracts
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-sm text-gray-400 transition-colors hover:text-white">
                    Audits
                  </Link>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="mb-4 text-lg font-medium text-white">Community</h3>
              <ul className="grid gap-2">
                <li>
                  <Link href="#" className="text-sm text-gray-400 transition-colors hover:text-white">
                    Discord
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-sm text-gray-400 transition-colors hover:text-white">
                    Governance Forum
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-sm text-gray-400 transition-colors hover:text-white">
                    Token Holders
                  </Link>
                </li>
              </ul>
            </div>
          </div>
          <div className="mt-8 border-t border-gray-800 pt-8 text-center">
            <p className="text-sm text-gray-400">© {new Date().getFullYear()} AlphaPulse. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
