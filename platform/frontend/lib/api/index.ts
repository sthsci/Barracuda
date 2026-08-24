import { createHttpApiClient } from "./client";
import { createMockApiClient } from "./mock";

export * from "./types";
export * from "./client";
export * from "./mock";
export * from "./csv";

export const apiClient =
  process.env.NEXT_PUBLIC_API_MODE === "mock"
    ? createMockApiClient()
    : createHttpApiClient({ baseUrl: process.env.NEXT_PUBLIC_API_URL ?? "/api/v1" });
