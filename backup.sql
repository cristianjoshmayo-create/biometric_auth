--
-- PostgreSQL database dump
--

\restrict hFEdpTx7DFVTC6HujICRjcsr2GZvCi6hUPWSs9223YYa4arl8Jn1f5cF7hPW3qK

-- Dumped from database version 18.2
-- Dumped by pg_dump version 18.2

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: auth_logs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.auth_logs (
    id integer NOT NULL,
    user_id integer,
    auth_method character varying(50),
    confidence_score double precision,
    result character varying(20),
    failed_attempts integer,
    attempted_at timestamp without time zone
);


--
-- Name: auth_logs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.auth_logs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: auth_logs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.auth_logs_id_seq OWNED BY public.auth_logs.id;


--
-- Name: keystroke_templates; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.keystroke_templates (
    id integer NOT NULL,
    user_id integer,
    dwell_times double precision[] NOT NULL,
    flight_times double precision[] NOT NULL,
    typing_speed double precision,
    dwell_mean double precision,
    dwell_std double precision,
    dwell_median double precision,
    dwell_min double precision,
    dwell_max double precision,
    flight_mean double precision,
    flight_std double precision,
    flight_median double precision,
    p2p_mean double precision,
    p2p_std double precision,
    r2r_mean double precision,
    r2r_std double precision,
    digraph_th double precision,
    digraph_he double precision,
    digraph_in double precision,
    digraph_er double precision,
    digraph_an double precision,
    digraph_ed double precision,
    digraph_to double precision,
    digraph_it double precision,
    typing_speed_cpm double precision,
    typing_duration double precision,
    rhythm_mean double precision,
    rhythm_std double precision,
    rhythm_cv double precision,
    pause_count double precision,
    pause_mean double precision,
    backspace_ratio double precision,
    backspace_count double precision,
    hand_alternation_ratio double precision,
    same_hand_sequence_mean double precision,
    finger_transition_ratio double precision,
    seek_time_mean double precision,
    seek_time_count double precision,
    enrolled_at timestamp without time zone,
    shift_lag_mean double precision,
    shift_lag_std double precision,
    shift_lag_count double precision,
    dwell_mean_norm double precision,
    dwell_std_norm double precision,
    flight_mean_norm double precision,
    flight_std_norm double precision,
    p2p_std_norm double precision,
    r2r_mean_norm double precision,
    shift_lag_norm double precision,
    digraph_bi double precision,
    digraph_io double precision,
    digraph_om double precision,
    digraph_me double precision,
    digraph_et double precision,
    digraph_tr double precision,
    digraph_ri double precision,
    digraph_ic double precision,
    digraph_vo double precision,
    digraph_oi double precision,
    digraph_ce double precision,
    digraph_ke double precision,
    digraph_ey double precision,
    digraph_ys double precision,
    digraph_st double precision,
    digraph_ro double precision,
    digraph_ok double precision,
    digraph_au double precision,
    digraph_ut double precision,
    digraph_he2 double precision,
    digraph_en double precision,
    digraph_nt double precision,
    digraph_ti double precision,
    digraph_ca double precision,
    digraph_at double precision,
    digraph_on double precision,
    attempt_number integer DEFAULT 1,
    enrollment_session integer DEFAULT 1,
    source character varying(20) DEFAULT 'enrollment'::character varying,
    sample_order integer DEFAULT 0
);


--
-- Name: keystroke_templates_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.keystroke_templates_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: keystroke_templates_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.keystroke_templates_id_seq OWNED BY public.keystroke_templates.id;


--
-- Name: security_questions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.security_questions (
    id integer NOT NULL,
    user_id integer,
    question text NOT NULL,
    answer_hash text NOT NULL
);


--
-- Name: security_questions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.security_questions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: security_questions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.security_questions_id_seq OWNED BY public.security_questions.id;


--
-- Name: users; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.users (
    id integer NOT NULL,
    username character varying(100) NOT NULL,
    is_flagged boolean,
    created_at timestamp without time zone,
    password_hash character varying(255)
);


--
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- Name: voice_templates; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.voice_templates (
    id integer NOT NULL,
    user_id integer,
    mfcc_features double precision[] NOT NULL,
    enrolled_at timestamp without time zone,
    mfcc_std double precision[] DEFAULT '{}'::double precision[],
    pitch_mean double precision DEFAULT 0,
    pitch_std double precision DEFAULT 0,
    speaking_rate double precision DEFAULT 0,
    energy_mean double precision DEFAULT 0,
    energy_std double precision DEFAULT 0,
    zcr_mean double precision DEFAULT 0,
    spectral_centroid_mean double precision DEFAULT 0,
    spectral_rolloff_mean double precision DEFAULT 0,
    attempt_number integer DEFAULT 1
);


--
-- Name: voice_templates_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.voice_templates_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: voice_templates_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.voice_templates_id_seq OWNED BY public.voice_templates.id;


--
-- Name: auth_logs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.auth_logs ALTER COLUMN id SET DEFAULT nextval('public.auth_logs_id_seq'::regclass);


--
-- Name: keystroke_templates id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.keystroke_templates ALTER COLUMN id SET DEFAULT nextval('public.keystroke_templates_id_seq'::regclass);


--
-- Name: security_questions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.security_questions ALTER COLUMN id SET DEFAULT nextval('public.security_questions_id_seq'::regclass);


--
-- Name: users id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- Name: voice_templates id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.voice_templates ALTER COLUMN id SET DEFAULT nextval('public.voice_templates_id_seq'::regclass);


--
-- Data for Name: auth_logs; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.auth_logs (id, user_id, auth_method, confidence_score, result, failed_attempts, attempted_at) FROM stdin;
\.


--
-- Data for Name: keystroke_templates; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.keystroke_templates (id, user_id, dwell_times, flight_times, typing_speed, dwell_mean, dwell_std, dwell_median, dwell_min, dwell_max, flight_mean, flight_std, flight_median, p2p_mean, p2p_std, r2r_mean, r2r_std, digraph_th, digraph_he, digraph_in, digraph_er, digraph_an, digraph_ed, digraph_to, digraph_it, typing_speed_cpm, typing_duration, rhythm_mean, rhythm_std, rhythm_cv, pause_count, pause_mean, backspace_ratio, backspace_count, hand_alternation_ratio, same_hand_sequence_mean, finger_transition_ratio, seek_time_mean, seek_time_count, enrolled_at, shift_lag_mean, shift_lag_std, shift_lag_count, dwell_mean_norm, dwell_std_norm, flight_mean_norm, flight_std_norm, p2p_std_norm, r2r_mean_norm, shift_lag_norm, digraph_bi, digraph_io, digraph_om, digraph_me, digraph_et, digraph_tr, digraph_ri, digraph_ic, digraph_vo, digraph_oi, digraph_ce, digraph_ke, digraph_ey, digraph_ys, digraph_st, digraph_ro, digraph_ok, digraph_au, digraph_ut, digraph_he2, digraph_en, digraph_nt, digraph_ti, digraph_ca, digraph_at, digraph_on, attempt_number, enrollment_session, source, sample_order) FROM stdin;
1	1	{196.10000002384186,143.60000002384186,155.40000000596046,146.2999999821186,136.19999998807907,89.2000000178814,99.90000000596046,131,130.5,111.2000000178814,133.2000000178814,176.80000001192093,163.2000000178814,124,165,155.2000000178814,149.80000001192093,128.19999998807907,98.59999999403954,120,101.90000000596046,127.90000000596046,173.2000000178814,138.2999999821186,125.69999998807907,121.80000001192093,150.30000001192093,108,97.80000001192093,108.2000000178814,138.40000000596046,125.90000000596046,104.19999998807907,126.40000000596046,154.19999998807907,162.40000000596046,137.19999998807907,297.59999999403954,181.39999997615814,129.90000000596046}	{99.2000000178814,0,96.5,120.59999999403954,33.599999994039536,37.29999998211861,4.5,54.19999998807907,0,0,18.19999998807907,0,106.5,0,119.7999999821186,130.5,0,0,59.400000005960464,2.5999999940395355,1.5,60.69999998807907,9.299999982118607,16.899999976158142,50.80000001192093,7.199999988079071,61.30000001192093,78.90000000596046,42.400000005960464}	6.906792830741932	139.10250000357627	35.42688588770849	132.1000000089407	89.2000000178814	297.59999999403954	41.789655169536324	42.98892676184767	33.599999994039536	145.16666666666666	53.14291005448787	143.46923076877226	59.94985577354845	107.09999999403954	125.09999999403954	\N	\N	\N	\N	\N	\N	414.40756984451593	5.791400000005961	145.16666666666666	53.14291005448787	0.3660820440033608	0	0	0	0	0.6666666666666666	1.625	0.6666666666666666	0	0	2026-03-09 15:25:16.847352	0	0	0	0.958226176832902	0.24404284193599418	0.28787362918165094	0.29613497195302646	0.3660820440033608	0.988306985777995	0	97	94.90000000596046	254.60000002384186	133.59999999403954	232.69999998807907	221.1000000089407	133.5	177.63333333532015	116.5	121	50.099999994039536	116.5	234.69999998807907	97.7000000178814	239.7999999821186	96.69999998807907	91.7000000178814	151.80000001192093	168.69999998807907	\N	189.2000000178814	133.09999999403954	117.5	84.39999997615814	204.80000001192093	74.89999997615814	1	1	enrollment	1
2	1	{200.7000000178814,200.7000000178814,152.7999999821186,152.7999999821186,126.40000000596046,126.40000000596046,121.5,121.40000000596046,134.30000001192093,134.19999998807907,106.30000001192093,106.30000001192093,81.40000000596046,81.40000000596046,132.7999999821186,132.7999999821186,131.69999998807907,131.69999998807907,114.30000001192093,114.2000000178814,134.2999999821186,134.2999999821186,202,202,179.40000000596046,179.2999999821186,171.40000000596046,171.40000000596046,167.69999998807907,167.59999999403954,185.40000000596046,185.2999999821186,126.2000000178814,126.2000000178814,141.7000000178814,141.7000000178814,98.7000000178814,98.59999999403954,111.30000001192093,111.19999998807907,99,99,118.90000000596046,118.90000000596046,162.5,162.40000000596046,140.59999999403954,140.59999999403954,127.09999999403954,127.09999999403954,135,135,157,157.10000002384186,118.80000001192093,118.80000001192093,109.7000000178814,109.7000000178814,110.89999997615814,110.89999997615814,152.5,152.5,124.80000001192093,124.90000000596046,121.2000000178814,121.2000000178814,109,109.09999999403954,118.09999999403954,118.09999999403954,140.39999997615814,140.39999997615814,104.7999999821186,104.90000000596046,294.2000000178814,294.30000001192093,181.59999999403954,181.59999999403954,131.90000000596046,131.80000001192093}	{111.69999998807907,0,88.5,106.09999999403954,47.099999994039536,44.30000001192093,13.900000005960464,31,4.100000023841858,1.9000000059604645,10.5,0,0,82.7999999821186,0,131,107,0,0,46.900000005960464,11.399999976158142,0,56.69999998807907,62.5,12.800000011920929,145,22.69999998807907,49.19999998807907,89,64.7000000178814,0}	13.301852282890849	139.4512500010431	37.43096977751778	131.75	81.40000000596046	294.30000001192093	43.251612902648986	44.063289311881455	31	74.46075949374634	84.48284885675871	73.5886075949367	85.97931012022926	172.2000000178814	123.69999998807907	\N	\N	\N	\N	\N	\N	798.1111369734509	6.0142000000178815	74.46075949374634	84.48284885675871	1.1345955833804526	0	0	0	0	0.3291139240506329	1.962962962962963	0.3291139240506329	0	0	2026-03-09 15:25:25.676175	0	0	0	1.872815304989671	0.502693902560334	0.5808645143658724	0.5917652413360376	1.1345955833804526	0.9882870937022489	0	143.60000002384186	94.79999999701977	238.09999999403954	99.09999999403954	222.69999998807907	209.20000000298023	128.5	185.4666666587194	138.40000000596046	117.39999997615814	92	107.95000000298023	224.5	88.89999997615814	242.19999998807907	112.80000001192093	88.5	146.40000000596046	175.5	\N	297.5	147.59999999403954	120.29999999701977	53.70000001788139	205.09999999403954	71.40000000596046	2	1	enrollment	2
3	1	{206,206,206,141.40000000596046,141.40000000596046,141.2999999821186,134,133.90000000596046,133.90000000596046,129.7999999821186,129.7999999821186,129.7999999821186,118.59999999403954,118.5,118.5,111.7999999821186,111.7999999821186,111.7999999821186,98.80000001192093,98.80000001192093,98.80000001192093,110.80000001192093,110.69999998807907,110.69999998807907,101.7000000178814,101.7000000178814,101.7000000178814,92.69999998807907,92.69999998807907,92.69999998807907,94.59999999403954,94.59999999403954,94.59999999403954,186.39999997615814,186.39999997615814,186.2999999821186,159.90000000596046,160,160,137.5,137.5,137.5,156.59999999403954,156.5,156.5,147.59999999403954,147.59999999403954,147.59999999403954,144.39999997615814,144.39999997615814,144.39999997615814,135.5,135.5,135.5,103,102.90000000596046,102.90000000596046,109.2999999821186,109.2999999821186,109.2999999821186,83,83,82.89999997615814,126.2000000178814,126.2000000178814,126.2000000178814,160.2999999821186,160.2999999821186,160.2999999821186,150.09999999403954,150.09999999403954,150.09999999403954,123.7000000178814,123.7000000178814,123.7000000178814,113.80000001192093,113.80000001192093,113.80000001192093,163.09999999403954,163.09999999403954,163.09999999403954,103.5,103.5,103.5,106.09999999403954,106.09999999403954,106.09999999403954,109.30000001192093,109.30000001192093,109.30000001192093,133.7000000178814,133.7000000178814,133.7000000178814,109.09999999403954,109.09999999403954,109.09999999403954,123,123,123,110.39999997615814,110.39999997615814,110.39999997615814,160.5,160.59999999403954,160.59999999403954,158.19999998807907,158.19999998807907,158.19999998807907,134.7999999821186,134.7999999821186,134.7999999821186,280.10000002384186,280.10000002384186,280.10000002384186,147.90000000596046,147.90000000596046,147.90000000596046,130.39999997615814,130.2999999821186,130.2999999821186}	{102.80000001192093,0,113.10000002384186,109.09999999403954,64.19999998807907,65.59999999403954,25,41.80000001192093,40.900000005960464,0,35.099999994039536,0,100.80000001192093,0,124.09999999403954,103.90000000596046,8.5,0,59.29999998211861,4.5,0.20000001788139343,16.69999998807907,11.099999994039536,12.199999988079071,31.5,17.30000001192093,33,43.80000001192093,0,25.80000001192093}	21.496130696566443	133.6808333305021	35.110751668693794	130.0499999821186	82.89999997615814	280.10000002384186	39.67666666805744	39.6730108385838	28.650000005960464	45.815966386504535	71.74004189095434	45.17983193257276	73.00179413041408	117.19999998807907	121.5	\N	\N	\N	\N	\N	\N	1289.7678417939865	5.582399999976158	45.815966386504535	71.74004189095434	1.565830594639295	0	0	0	0	0.2184873949579832	3.4444444444444446	0.2184873949579832	0	0	2026-03-09 15:25:33.884245	0	0	0	2.9177783177761123	0.7663431427485059	0.8660008681983085	0.8659210744110771	1.565830594639295	0.9861154417530926	0	147.89999997615814	100.70000000298023	236.7000000178814	94.2999999821186	231.60000002384186	203.84999997913837	163	153.9999999900659	135.5	106.09999999403954	72.90000000596046	120.14999997615814	236.30000001192093	58.70000001788139	233.39999997615814	134.7000000178814	81	163.30000001192093	120.19999998807907	\N	165.2000000178814	126.40000000596046	114.6499999910593	114.90000000596046	184	41.400000005960464	3	1	enrollment	3
\.


--
-- Data for Name: security_questions; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.security_questions (id, user_id, question, answer_hash) FROM stdin;
\.


--
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.users (id, username, is_flagged, created_at, password_hash) FROM stdin;
1	josh	f	2026-03-09 15:25:05.088877	$2b$12$S/5/tc5ml1/i1V8rM1iItuaB8RQ6bfhHkAxQYz15ADQ/WTtUmZKki
\.


--
-- Data for Name: voice_templates; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.voice_templates (id, user_id, mfcc_features, enrolled_at, mfcc_std, pitch_mean, pitch_std, speaking_rate, energy_mean, energy_std, zcr_mean, spectral_centroid_mean, spectral_rolloff_mean, attempt_number) FROM stdin;
\.


--
-- Name: auth_logs_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.auth_logs_id_seq', 1, false);


--
-- Name: keystroke_templates_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.keystroke_templates_id_seq', 3, true);


--
-- Name: security_questions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.security_questions_id_seq', 1, false);


--
-- Name: users_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.users_id_seq', 1, true);


--
-- Name: voice_templates_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.voice_templates_id_seq', 1, false);


--
-- Name: auth_logs auth_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.auth_logs
    ADD CONSTRAINT auth_logs_pkey PRIMARY KEY (id);


--
-- Name: keystroke_templates keystroke_templates_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.keystroke_templates
    ADD CONSTRAINT keystroke_templates_pkey PRIMARY KEY (id);


--
-- Name: security_questions security_questions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.security_questions
    ADD CONSTRAINT security_questions_pkey PRIMARY KEY (id);


--
-- Name: security_questions security_questions_user_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.security_questions
    ADD CONSTRAINT security_questions_user_id_key UNIQUE (user_id);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: users users_username_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_username_key UNIQUE (username);


--
-- Name: voice_templates voice_templates_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.voice_templates
    ADD CONSTRAINT voice_templates_pkey PRIMARY KEY (id);


--
-- Name: ix_users_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_users_id ON public.users USING btree (id);


--
-- Name: auth_logs auth_logs_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.auth_logs
    ADD CONSTRAINT auth_logs_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);


--
-- Name: keystroke_templates keystroke_templates_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.keystroke_templates
    ADD CONSTRAINT keystroke_templates_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);


--
-- Name: security_questions security_questions_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.security_questions
    ADD CONSTRAINT security_questions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);


--
-- Name: voice_templates voice_templates_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.voice_templates
    ADD CONSTRAINT voice_templates_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);


--
-- PostgreSQL database dump complete
--

\unrestrict hFEdpTx7DFVTC6HujICRjcsr2GZvCi6hUPWSs9223YYa4arl8Jn1f5cF7hPW3qK

