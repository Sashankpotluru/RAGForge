import * as runtime from "@huggingface/transformers";
import { modelCache } from "./dataset.mjs";
import { createModels } from "../web/neural.mjs";

runtime.env.cacheDir = modelCache;
runtime.env.allowLocalModels = false;
export const session = { device: "cpu", session_options: { intraOpNumThreads: 6, interOpNumThreads: 1 } };
export const neural = createModels(runtime, session);
