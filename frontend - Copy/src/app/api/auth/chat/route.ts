import { NextResponse } from "next/server";

export async function POST(req: Request) {
  console.log("DEBUG: Frontend API route handler called");

  try {
    const requestData = await req.json();
    console.log("DEBUG: Request data:", requestData);

    const backendUrl = "http://localhost:8000/api/chat"; // Your FastAPI backend
    console.log("DEBUG: Backend URL:", backendUrl);

    const response = await fetch(backendUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestData),
    });

    console.log("DEBUG: Backend response status:", response.status);
    console.log(
      "DEBUG: Backend response headers:",
      Object.fromEntries(response.headers.entries())
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error("DEBUG: Backend error response:", errorText);
      let errorData;
      try {
        errorData = JSON.parse(errorText);
      } catch {
        errorData = { detail: errorText };
      }
      return NextResponse.json(
        { error: errorData.detail || "Backend error" },
        { status: response.status }
      );
    }

    const data = await response.json();
    console.log("DEBUG: Backend success response:", data);
    return NextResponse.json(data);
  } catch (error) {
    console.error("DEBUG: Error in route handler:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
