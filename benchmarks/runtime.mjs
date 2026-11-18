import * as runtime from "@huggingface/transformers";
import { join } from "node:path";
import { rawDir } from "./dataset.mjs";
import { createModels } from "../web/neural.mjs";

runtime.env.cacheDir = join(rawDir, "models");
runtime.env.allowLocalModels = false;
export const session = { device: "cpu", session_options: { intraOpNumThreads: 6, interOpNumThreads: 1 } };
export const neural = createModels(runtime, session);
