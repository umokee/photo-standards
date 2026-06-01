import { client } from "@/lib/api-client";
import { MutationConfig } from "@/lib/react-query";
import { useMutation } from "@tanstack/react-query";

export type SamPromptPoint = {
  x: number;
  y: number;
  label: 0 | 1;
};

export type SamPredictInput = {
  imageId: string;
  positive: [number, number][];
  negative: [number, number][];
};

export type SamPredictResponse = {
  points: number[][];
  score: number;
};

export const samPredict = async ({
  imageId,
  positive,
  negative,
}: SamPredictInput): Promise<SamPredictResponse> => {
  const points: SamPromptPoint[] = [
    ...positive.map(([x, y]) => ({ x, y, label: 1 as const })),
    ...negative.map(([x, y]) => ({ x, y, label: 0 as const })),
  ];

  return client.post(`/sam/predict`, {
    image_id: imageId,
    points,
  });
};

type Options = {
  mutationConfig?: MutationConfig<typeof samPredict>;
};

export const samWarmup = ({ imageId }: { imageId: string }) =>
  client.post(`/sam/warmup/${imageId}`);

export const useSamWarmup = () => useMutation({ mutationFn: samWarmup });

export const useSamPredict = ({ mutationConfig }: Options = {}) => {
  return useMutation({
    mutationFn: samPredict,
    ...mutationConfig,
  });
};
