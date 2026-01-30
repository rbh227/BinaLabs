import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "DA-SegFormer | Bina Labs — Lehigh University",
  description:
    "Damage-Aware Semantic Segmentation for Fine-Grained Post-Disaster Assessment",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} font-sans antialiased bg-[#0A0B0F] text-gray-100`}>
        {children}
      </body>
    </html>
  );
}
