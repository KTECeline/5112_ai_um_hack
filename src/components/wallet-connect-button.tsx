"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Wallet, Copy, Check, ExternalLink } from "lucide-react"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

export function WalletConnectButton() {
  const [isConnected, setIsConnected] = useState(false)
  const [isCopied, setIsCopied] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)

  const mockAddress = "0x71C7656EC7ab88b098defB751B7401B5f6d8976F"
  const shortenedAddress = `${mockAddress.substring(0, 6)}...${mockAddress.substring(mockAddress.length - 4)}`

  const handleConnect = () => {
    setIsConnecting(true)
    // Simulate connection delay
    setTimeout(() => {
      setIsConnected(true)
      setIsConnecting(false)
    }, 1500)
  }

  const handleDisconnect = () => {
    setIsConnected(false)
  }

  const handleCopyAddress = () => {
    navigator.clipboard.writeText(mockAddress)
    setIsCopied(true)
    setTimeout(() => setIsCopied(false), 2000)
  }

  if (isConnected) {
    return (
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="outline"
            className="border-purple-800/50 bg-purple-500/10 text-purple-300 hover:bg-purple-500/20 hover:text-purple-200"
          >
            <div className="mr-2 h-2 w-2 rounded-full bg-green-400"></div>
            {shortenedAddress}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="w-56 bg-gray-900 border border-gray-800">
          <DropdownMenuLabel className="text-gray-400">Connected Wallet</DropdownMenuLabel>
          <DropdownMenuSeparator className="bg-gray-800" />
          <DropdownMenuItem
            className="flex cursor-pointer items-center justify-between text-gray-300 focus:bg-gray-800 focus:text-white"
            onClick={handleCopyAddress}
          >
            <span>{shortenedAddress}</span>
            {isCopied ? <Check className="h-4 w-4 text-green-400" /> : <Copy className="h-4 w-4" />}
          </DropdownMenuItem>
          <DropdownMenuItem className="cursor-pointer text-gray-300 focus:bg-gray-800 focus:text-white">
            <ExternalLink className="mr-2 h-4 w-4" />
            <span>View on Etherscan</span>
          </DropdownMenuItem>
          <DropdownMenuSeparator className="bg-gray-800" />
          <DropdownMenuItem
            className="cursor-pointer text-red-400 focus:bg-gray-800 focus:text-red-300"
            onClick={handleDisconnect}
          >
            Disconnect
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    )
  }

  return (
    <Button
      variant="outline"
      className="border-purple-800/50 bg-purple-500/10 text-purple-300 hover:bg-purple-500/20 hover:text-purple-200"
      onClick={handleConnect}
      disabled={isConnecting}
    >
      {isConnecting ? (
        <>
          <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-purple-400 border-t-transparent"></div>
          Connecting...
        </>
      ) : (
        <>
          <Wallet className="mr-2 h-4 w-4" />
          Connect Wallet
        </>
      )}
    </Button>
  )
}
