import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { queryOptions, useQuery } from "@tanstack/react-query";

export type AppConstantsResponse = {
  users: {
    roles: {
      values: string[];
      default: string;
    };
  };
  standards: {
    angles: {
      values: string[];
    };
  };
  inspections: {
    statuses: {
      values: string[];
      default: string;
    };
    modes: {
      values: string[];
      default: string;
    };
  };
  training: {
    statuses: {
      values: string[];
      active: string[];
      default: string;
    };
    architectures: {
      values: string[];
      default: string;
    };
    image_size: {
      values: number[];
      default: number;
    };
    epochs: {
      default: number;
      min: number;
      max: number;
    };
    batch_size: {
      default: number;
      min: number;
      max: number;
    };
    train_ratio: {
      default: number;
      min: number;
      max: number;
    };
    val_ratio: {
      default: number;
      min: number;
      max: number;
    };
    ratio_sum_max: number;
    min_images_to_train: number;
  };
  segments: {
    hue: {
      default: number;
      min: number;
      max: number;
    };
  };
  uploads: {
    allowed_types: {
      values: string[];
    };
    max_size_bytes: number;
  };
};

export const getConstants = (): Promise<AppConstantsResponse> => {
  return client.get("/meta/constants");
};

export const getConstantsQueryOptions = () =>
  queryOptions({
    queryKey: queryKeys.meta.constants(),
    queryFn: getConstants,
    staleTime: Infinity,
    gcTime: Infinity,
  });

export const useGetConstants = () => {
  return useQuery(getConstantsQueryOptions());
};
