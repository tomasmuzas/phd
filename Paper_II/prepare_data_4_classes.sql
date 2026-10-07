create table "AllGalaxiesNew"
(
    "OriginalDatasetId"             text,
    "OriginalDataset"               text,
    "ra"                            float8,
    "dec"                           float8,
    "Spec_z"                        float8,
    "Spec_z_err"                    float8,
    "Spec_mag"                      float8,
    "Spec_mag_err"                  float8,
    "Features"                      float8,
    "Smooth"                        float8,
    "Artifact"                      float8,
    "Features_Clumpy_Yes"           float8,
    "Features_Clumpy_No"            float8,
    "Features_EdgeOn_Yes"           float8,
    "Features_EdgeOn_No"            float8,
    "Features_Flat_Bar_Yes"         float8,
    "Features_Flat_Bar_No"          float8
);

create unique index id_per_dataset_new
    on "AllGalaxies" ("OriginalDatasetId", "OriginalDataset");

create index original_id_new
    on "AllGalaxies" ("OriginalDatasetId");

UPDATE "AllGalaxies" AS g
SET ra = c.ra, dec = c.dec
FROM "gz1_coords" AS c
WHERE g."OriginalDataset" = 'GZ1'
  AND g."OriginalDatasetId" = c.originaldatasetid;


ALTER TABLE "AllGalaxies"
ADD COLUMN "Id" uuid;

UPDATE "AllGalaxies"
SET "Id" = gen_random_uuid();

SELECT *, st_point(ra, dec) AS "Coordinates" 
INTO "AllGalaxiesWithCoords"
FROM "AllGalaxies";

SELECT "Id", "OriginalDatasetId", "OriginalDataset", "ra", "dec", "Coordinates", "Features",
       "Smooth"              ,
       "Artifact"            ,
       "Features_Clumpy_Yes" ,
       "Features_Clumpy_No"  ,
       "Features_EdgeOn_Yes" ,
       "Features_EdgeOn_No"  ,
       "Features_Flat_Bar_Yes",
       "Features_Flat_Bar_No"
INTO "FourClassificationInitial"
FROM "AllGalaxies";

-- Calculate class
ALTER TABLE "FourClassificationInitial"
ADD COLUMN "Class" text;

UPDATE "FourClassificationInitial"
SET "Class" = CASE 
    WHEN "OriginalDataset" = 'GZ2' THEN
        CASE 
            WHEN "Smooth" >= 0.8 AND "Features" < 0.8 THEN 'Elliptical'
            WHEN "Features" >= 0.8 AND "Smooth" < 0.8 AND "Features_EdgeOn_Yes" >= 0.8 THEN 'EdgeOn'
            WHEN "Features" >= 0.8 AND "Smooth" < 0.8 AND "Features_EdgeOn_No" >= 0.8 AND "Features_Flat_Bar_Yes" >= 0.8 THEN 'SpiralBar'
            WHEN "Features" >= 0.8 AND "Smooth" < 0.8 AND "Features_EdgeOn_No" >= 0.8 AND "Features_Flat_Bar_No" >= 0.8 THEN 'SpiralNoBar'
            ELSE 'Uncertain'
        END
    WHEN "OriginalDataset" = 'GZ_HUBBLE' THEN
        CASE
            WHEN "Smooth" >= 0.8 AND "Features" < 0.8 THEN 'Elliptical'
            WHEN "Features" >= 0.8 AND "Smooth" < 0.8 AND "Features_Clumpy_No" >= 0.8 AND "Features_EdgeOn_Yes" >= 0.8 THEN 'EdgeOn'
            WHEN "Features" >= 0.8 AND "Smooth" < 0.8 AND "Features_Clumpy_No" >= 0.8 AND "Features_EdgeOn_No" >= 0.8 AND "Features_Flat_Bar_Yes" >= 0.8 THEN 'SpiralBar'
            WHEN "Features" >= 0.8 AND "Smooth" < 0.8 AND "Features_Clumpy_No" >= 0.8 AND "Features_EdgeOn_No" >= 0.8 AND "Features_Flat_Bar_No" >= 0.8 THEN 'SpiralNoBar'
            ELSE 'Uncertain'
        END
    WHEN "OriginalDataset" IN ('GZ_DECALS_12', 'GZ_DECALS_5') THEN
        CASE
            WHEN "Smooth" >= 0.8 AND "Features" < 0.8 THEN 'Elliptical'
            WHEN "Features" >= 0.8 AND "Smooth" < 0.8 AND "Features_EdgeOn_Yes" >= 0.8 THEN 'EdgeOn'
            WHEN "Features" >= 0.8 AND "Smooth" < 0.8 AND "Features_EdgeOn_No" >= 0.8 AND "Features_Flat_Bar_No" < 0.2 THEN 'SpiralBar'
            WHEN "Features" >= 0.8 AND "Smooth" < 0.8 AND "Features_EdgeOn_No" >= 0.8 AND "Features_Flat_Bar_No" >= 0.8 THEN 'SpiralNoBar'
            ELSE 'Uncertain'
        END
    WHEN "OriginalDataset" = 'GZ_CANDELS' THEN
        CASE
            WHEN "Smooth" >= 0.8 AND "Features" < 0.8 THEN 'Elliptical'
            WHEN "Features" >= 0.8 AND "Smooth" < 0.8 AND "Features_Clumpy_No" >= 0.8 AND "Features_EdgeOn_Yes" >= 0.8 THEN 'EdgeOn'
            WHEN "Features" >= 0.8 AND "Smooth" < 0.8 AND "Features_Clumpy_No" >= 0.8 AND "Features_EdgeOn_No" >= 0.8 AND "Features_Flat_Bar_Yes" >= 0.8 THEN 'SpiralBar'
            WHEN "Features" >= 0.8 AND "Smooth" < 0.8 AND "Features_Clumpy_No" >= 0.8 AND "Features_EdgeOn_No" >= 0.8 AND "Features_Flat_Bar_No" >= 0.8 THEN 'SpiralNoBar'
            ELSE 'Uncertain'
            END
    END;

DELETE FROM "FourClassificationInitial"
WHERE "OriginalDataset" = 'GZ1';

SELECT *
INTO "FourClassificationClean"
FROM "FourClassificationInitial";

-- Step 3: crossmatch by coordinates
ALTER TABLE "FourClassificationClean"
    ADD COLUMN "Coordinates_GZ1_Id" text,
    ADD COLUMN "Coordinates_GZ1_Class" text,
    ADD COLUMN "Coordinates_GZ2_Id" text,
    ADD COLUMN "Coordinates_GZ2_Class" text,
    ADD COLUMN "Coordinates_GZD_Id" text,
    ADD COLUMN "Coordinates_GZD_Class" text,
    ADD COLUMN "Coordinates_GZH_Id" text,
    ADD COLUMN "Coordinates_GZH_Class" text,
    ADD COLUMN "Coordinates_GZC_Id" text,
    ADD COLUMN "Coordinates_GZC_Class" text;

CREATE INDEX IDX_coordinates_four_classes ON "FourClassificationClean" USING GIST("Coordinates");

-- GZ1
WITH gz1 As (
    SELECT "OriginalDatasetId", "Class", "Coordinates", ra, dec
    FROM "FourClassificationClean"
    WHERE "OriginalDataset" IN ('GZ1')
),
     sub AS (
         SELECT main."Id", gz1."OriginalDatasetId", gz1."Class"
         FROM "FourClassificationClean" AS main
                  LEFT JOIN gz1 ON st_dwithin(gz1."Coordinates", main."Coordinates", 0.000277778 / 2) -- 0.5 arcsec
         WHERE gz1."OriginalDatasetId" IS NOT NULL
     )
UPDATE "FourClassificationClean" AS main
SET "Coordinates_GZ1_Id" = sub."OriginalDatasetId", "Coordinates_GZ1_Class" = sub."Class"
FROM sub
WHERE main."Id" = sub."Id";


-- GZ2
WITH gz2 As (
    SELECT "OriginalDatasetId", "Class", "Coordinates", ra, dec
    FROM "FourClassificationClean"
    WHERE "OriginalDataset" IN ('GZ2')
),
     sub AS (
         SELECT main."Id", gz2."OriginalDatasetId", gz2."Class"
         FROM "FourClassificationClean" AS main
                  LEFT JOIN gz2 ON st_dwithin(gz2."Coordinates", main."Coordinates", 0.000277778 / 2) -- 0.5 arcsec
         WHERE gz2."OriginalDatasetId" IS NOT NULL
     )
UPDATE "FourClassificationClean" AS main
SET "Coordinates_GZ2_Id" = sub."OriginalDatasetId", "Coordinates_GZ2_Class" = sub."Class"
FROM sub
WHERE main."Id" = sub."Id";

-- GZH

WITH gzh As (
    SELECT "OriginalDatasetId", "Class", "Coordinates", ra, dec
    FROM "FourClassificationClean"
    WHERE "OriginalDataset" = 'GZ_HUBBLE'
),
     sub AS (
         SELECT main."Id", gzh."OriginalDatasetId", gzh."Class"
         FROM "FourClassificationClean" AS main
                  LEFT JOIN gzh
                            ON st_dwithin(gzh."Coordinates", main."Coordinates", 0.000277778 / 2) -- 0.5 arcsec
         WHERE gzh."OriginalDatasetId" IS NOT NULL
     )
UPDATE "FourClassificationClean" AS main
SET "Coordinates_GZH_Id" = sub."OriginalDatasetId", "Coordinates_GZH_Class" = sub."Class"
FROM sub
WHERE main."Id" = sub."Id";

-- GZC

WITH gzc As (
    SELECT "OriginalDatasetId", "Class", "Coordinates", ra, dec
    FROM "FourClassificationClean"
    WHERE "OriginalDataset" = 'GZ_CANDELS'
),
     sub AS (
         SELECT main."Id", gzc."OriginalDatasetId", gzc."Class"
         FROM "FourClassificationClean" AS main
                  LEFT JOIN gzc
                            ON st_dwithin(gzc."Coordinates", main."Coordinates", 0.000277778 / 2) -- 0.5 arcsec
         WHERE gzc."OriginalDatasetId" IS NOT NULL
     )
UPDATE "FourClassificationClean" AS main
SET "Coordinates_GZC_Id" = sub."OriginalDatasetId", "Coordinates_GZC_Class" = sub."Class"
FROM sub
WHERE main."Id" = sub."Id";


WITH gzd As (
    SELECT "OriginalDatasetId", "Class", "Coordinates", ra, dec
    FROM "FourClassificationClean"
    WHERE "OriginalDataset" IN ('GZ_DECALS_12', 'GZ_DECALS_5')
),
     sub AS (
         SELECT main."Id", gzd."OriginalDatasetId", gzd."Class"
         FROM "FourClassificationClean" AS main
                  LEFT JOIN gzd
                            ON st_dwithin(gzd."Coordinates", main."Coordinates", 0.000277778 / 2) -- 0.5 arcsec
         WHERE gzd."OriginalDatasetId" IS NOT NULL
     )
UPDATE "FourClassificationClean" AS main
SET "Coordinates_GZD_Id" = sub."OriginalDatasetId", "Coordinates_GZD_Class" = sub."Class"
FROM sub
WHERE main."Id" = sub."Id";

-- Match classes
SELECT
    main.*,
    CASE WHEN
             (
                 WITH z as (SELECT DISTINCT UNNEST(array["Coordinates_GZ1_Class", "Coordinates_GZ2_Class", "Coordinates_GZH_Class", "Coordinates_GZC_Class", "Coordinates_GZD_Class"]) AS "Class")
                 SELECT COUNT(*) FROM z
                 WHERE z."Class" IS NOT NULL
             ) = 1
             THEN true
         ELSE false END AS "AllCoordinateClassesMatch"
INTO "FourClassificationCrossMatched"
FROM "FourClassificationClean" AS main

-- Delete non-matching entries (should take care of multiple entries, since they all have not matching entries)
DELETE FROM "FourClassificationCrossMatched"
WHERE "AllCoordinateClassesMatch" = false


DELETE
FROM "FourClassificationCrossMatched"
WHERE ctid IN (
    SELECT ctid
    FROM (
             SELECT ctid, "Coordinates_GZ1_Id", "Coordinates_GZ2_Id", "Coordinates_GZH_Id", "Coordinates_GZC_Id", "Coordinates_GZD_Id", row_number()
                                                                                                                                        OVER (PARTITION BY "Coordinates_GZ1_Id", "Coordinates_GZ2_Id", "Coordinates_GZH_Id", "Coordinates_GZC_Id", "Coordinates_GZD_Id" ORDER BY ctid) rn
             FROM "FourClassificationCrossMatched"
         ) t
    WHERE rn > 1
);

DELETE FROM "FourClassificationCrossMatched"
WHERE "Class" = 'Uncertain'

SELECT "Class", COUNT(*)
FROM "FourClassificationCrossMatched"
GROUP BY "Class"


SELECT *
INTO "FourClassificationFinal"
FROM "FourClassificationCrossMatched";

ALTER TABLE "FourClassificationFinal"
RENAME COLUMN "Coordinates_GZ2_Id" TO "GZ12_Id";

ALTER TABLE "FourClassificationFinal"
RENAME COLUMN "Coordinates_GZC_Id" TO "GZC_Id";

ALTER TABLE "FourClassificationFinal"
RENAME COLUMN "Coordinates_GZD_Id" TO "GZD_Id";

ALTER TABLE "FourClassificationFinal"
RENAME COLUMN "Coordinates_GZH_Id" TO "GZH_Id";

ALTER TABLE "AllGalaxies"
ADD COLUMN "Smooth_Round" double precision,
ADD COLUMN "Smooth_Cigar" double precision,
ADD COLUMN "Smooth_InBetween" double precision,
ADD COLUMN "Features_Flat_Spirals_Yes" double precision,
ADD COLUMN "Features_Flat_Spirals_No" double precision,
ADD COLUMN "Features_Flat_Spirals_Tight" double precision,
ADD COLUMN "Features_Flat_Spirals_Medium" double precision,
ADD COLUMN "Features_Flat_Spirals_Loose" double precision;

UPDATE "AllGalaxies" AS g
SET
    "Smooth_Cigar" = o."smooth_cigar",
    "Smooth_Round" = o."smooth_round",
    "Smooth_InBetween" = o."smooth_inbetween",
    "Features_Flat_Spirals_Yes" = o."features_flat_spirals_yes",
    "Features_Flat_Spirals_No" = o."features_flat_spirals_no",
    "Features_Flat_Spirals_Tight" = o."features_flat_spirals_tight",
    "Features_Flat_Spirals_Medium" = o."features_flat_spirals_medium",
    "Features_Flat_Spirals_Loose" = o."features_flat_spirals_loose"
FROM "gzd_12_data" AS o
WHERE g."OriginalDataset" = 'GZ_DECALS_12' AND g."OriginalDatasetId" = o.originaldatasetid
