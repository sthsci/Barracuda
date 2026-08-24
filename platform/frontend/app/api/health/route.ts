import { NextResponse } from "next/server";

export function GET() {
  return NextResponse.json({
    status: "ok",
    service: "barracuda-frontend",
    apiMode: process.env.NEXT_PUBLIC_API_MODE ?? "mock",
  });
}
