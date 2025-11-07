-- Enable pgcrypto for gen_random_uuid (usually already on)
create extension if not exists pgcrypto;

-- 1) Lobbies (unique code)
create table if not exists lobbies (
  id uuid primary key default gen_random_uuid(),
  code text unique not null,
  created_at timestamptz not null default now()
);

-- 2) Draft snapshot per lobby (single row)
create table if not exists drafts (
  lobby_id uuid primary key references lobbies(id) on delete cascade,
  state jsonb not null,
  step_idx int not null default 0,
  updated_at timestamptz not null default now()
);

-- Updated_at trigger
create or replace function set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end $$;

drop trigger if exists trg_drafts_updated_at on drafts;
create trigger trg_drafts_updated_at
before update on drafts
for each row execute function set_updated_at();

-- Turn OFF RLS for quick prototype (early dev)
alter table lobbies disable row level security;
alter table drafts  disable row level security;
